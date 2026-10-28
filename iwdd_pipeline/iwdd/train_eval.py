"""Addestramento e valutazione. Tutte le scelte (feature, aggregazioni, C, delta, soglia) si fanno in CV a gruppi sul solo TRAIN;
il TEST serve solo a misurare. Due modelli sugli stessi segnali di Visual-JEV:
  1. classificatore di VIDEO (regressione logistica L2 su max/media/... dei segnali) -> c'e' uno scarico?
  2. modello a FINESTRE (L2 su una griglia) -> probabilita' che rispondere col tempo di quella griglia (+delta) sia un TP.
Risposta = tempo della griglia col punteggio massimo + delta, intero, entro il video; si risponde solo se il punteggio di video >= soglia.

Il tipo di modello e le griglie di iperparametri vengono da una specifica: senza classificatore e' la regressione logistica L2
originale (griglie di C in config/default.json); con --classifier NOME e' classifiers/NOME/model.json (vedi models.py). Le griglie
sono ordinate dal modello piu' semplice al piu' complesso: a parita' di risultato vince il primo.
"""
import json
import logging
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import cv as cvm
from .evaluate import answer_int, bootstrap, from_outcomes, grid_search, outcome
from .features import base_columns, read_features, video_matrix, window_matrix, window_probabilities
from .models import legacy_spec, make_model
from .scan import load_groups, load_meta

log = logging.getLogger("iwdd.train")


# ----------------------------------------------------------------------------------------------------------------------
def _info_table(videos, meta):
    rows = {}
    for v in videos:
        m = meta.get(v.vid)
        rows[v.vid] = dict(split=v.split, is_pos=v.is_pos, g=v.g, event_type=v.event_type,
                           dur=(m["t_last"] if m else v.duration), has_meta=m is not None)
    return pd.DataFrame.from_dict(rows, orient="index")


def _f1_detect(y, p, thr):
    pred = np.asarray(p) >= thr
    y = np.asarray(y).astype(bool)
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def _best_thr_detect(y, p):
    ts = np.unique(np.round(p, 4))
    return float(ts[int(np.argmax([_f1_detect(y, p, t) for t in ts]))])


def _shortcut_features(info, meta, vids):
    rows = {}
    for v in vids:
        m = meta.get(v)
        if not m:
            continue
        rows[v] = {"log_dur": np.log(max(m["t_last"], 0.5)), "log_px": np.log(m["width"] * m["height"]),
                   "fps": m["fps_mean"] or 0.0, "bright": m["brightness"], "night": float(info.loc[v, "night"])}
    return pd.DataFrame.from_dict(rows, orient="index")


def _question_stats(X, info, tr_ids, te_ids, cols):
    """AUC e correlazione di ogni segnale (aggregato con max e media) con l'etichetta, su train e test."""
    out = []
    Fm = {a: video_matrix(X, tr_ids + te_ids, (a,), cols) for a in ("max", "mean")}
    for c in cols:
        r = {"segnale": c}
        for a in ("max", "mean"):
            for name, ids in (("train", tr_ids), ("test", te_ids)):
                if not ids:
                    continue
                x = Fm[a].loc[ids, f"{c}|{a}"].to_numpy(dtype=float)
                y = info.loc[ids, "is_pos"].to_numpy(dtype=int)
                ok = ~np.isnan(x)
                r[f"AUC {a} {name}"] = roc_auc_score(y[ok], x[ok]) if len(set(y[ok])) == 2 else np.nan
                r[f"r {a} {name}"] = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 and x[ok].std() > 0 else np.nan
        out.append(r)
    return pd.DataFrame(out).set_index("segnale")


# ----------------------------------------------------------------------------------------------------------------------
def run_train_eval(run, cfg, questions, derived, train_videos, test_videos, time_col=None, model_tag="default", allow_partial=False,
                   spec=None, n_jobs=1):
    """spec: specifica del classificatore (None = logistica L2 originale). n_jobs: processi per gli addestramenti della CV."""
    spec = spec or legacy_spec(cfg)
    time_col = time_col or cfg["window_model"]["time_col"]
    clip, clip_d = cfg["clip_logit"], cfg["clip_derived"]
    excl = set(cfg["exclude_from_training"])
    all_videos = list(train_videos) + list(test_videos)
    meta = load_meta(run, [v.vid for v in all_videos])
    info = _info_table(all_videos, meta)
    info["night"] = [v.time_of_day == "night" for v in all_videos]

    long, wins, have = read_features(run, [v.vid for v in all_videos])
    if long.empty:
        raise RuntimeError(f"Nessuna feature trovata in {run.features}: esegui prima 'extract'.")
    X_all = window_matrix(long, questions, derived, clip, clip_d)
    W_idx = wins.set_index(["video", "win"]).reindex(X_all.index)
    Tw = W_idx[["t_start", "t_center", "t_end"]]
    # griglie di copertura (preprocessing_pt2.py, frame intero) vs griglie evento (zoom sul soggetto): Visual-JEV risponde in modo
    # diverso, quindi il modello a finestre riceve l'indicatore is_fill. NON entra nel classificatore di video: aggregato per video
    # misurerebbe quanto del video e' "vuoto", cioe' la durata, una scorciatoia (vedi baseline_shortcuts).
    is_fill = (W_idx["kind"] == "fill").astype(float) if "kind" in W_idx.columns else None
    if is_fill is not None and not is_fill.any():
        is_fill = None

    tr = [v for v in train_videos if v.vid in have and v.vid not in excl]
    te = [v for v in test_videos if v.vid in have]
    skipped = [v.vid for v in train_videos if v.vid not in have]
    skipped_te = [v.vid for v in test_videos if v.vid not in have]
    if (skipped or skipped_te) and not allow_partial:
        raise RuntimeError(f"Feature mancanti: {len(skipped)} video di train e {len(skipped_te)} di test (es. {(skipped + skipped_te)[:5]}). "
                           f"Completa l'estrazione (python -m iwdd status --run {run.name}) oppure usa --allow-partial.")
    if skipped or skipped_te:
        log.warning("%d video di train e %d di test senza feature (saltati)", len(skipped), len(skipped_te))
    if len(tr) < 20 or len({v.is_pos for v in tr}) < 2:
        raise RuntimeError(f"Troppo pochi video di train con feature ({len(tr)}).")
    tr_ids, te_ids = [v.vid for v in tr], [v.vid for v in te]
    y_tr = np.array([v.is_pos for v in tr]).astype(int)
    groups = load_groups(run)
    if not groups:
        log.warning("gruppi di camera assenti (esegui 'scan'): ogni video e' un gruppo -> la CV puo' essere ottimistica")
    gid = {}
    g_tr = np.array([gid.setdefault(groups.get(v, f"solo_{v}"), len(gid)) for v in tr_ids])
    folds = cvm.make_folds(y_tr, g_tr, cfg["cv"]["n_splits"], cfg["cv"]["repeats"], cfg["cv"]["seed"])
    log.info("train: %d video (%d positivi, %d senza griglie) | test: %d video | griglie totali %d | segnali per griglia %d",
             len(tr), y_tr.sum(), sum(1 for v in tr_ids if v not in set(X_all.index.get_level_values('video'))), len(te),
             len(X_all), X_all.shape[1])

    # ---------------- 1. classificatore di video: scelta di insieme di feature, aggregazioni e C (AUC out-of-fold)
    bcols = base_columns(X_all, derived)
    fsets = {"base": bcols, "derived": bcols + [d for d in derived if d in X_all.columns]}
    if fsets["derived"] == fsets["base"]:
        fsets.pop("derived")
    vm = cfg["video_model"]
    log.info("classificatore: %s (%s) | griglia video %d, griglia finestre %d | %d processi",
             spec["name"] or "logistica L2 originale", spec["kind"], len(spec["video_grid"]), len(spec["window_grid"]), n_jobs)
    grid, configs = [], []
    for fname in vm["feature_sets"]:
        if fname not in fsets:
            continue
        for aggs in vm["aggs_options"]:
            Fv = video_matrix(X_all, tr_ids, aggs, fsets[fname])
            for gi, params in enumerate(spec["video_grid"]):
                configs.append((Fv, partial(make_model, spec, params)))
                grid.append(dict(feature_set=fname, aggs="+".join(aggs), grid_idx=gi, params=json.dumps(params, sort_keys=True),
                                 **params, n_features=Fv.shape[1]))
    for row, oof in zip(grid, cvm.oof_video_many(configs, y_tr, folds, n_jobs)):
        row["auc_oof"] = roc_auc_score(y_tr, oof)
    G = pd.DataFrame(grid)
    G.to_csv(run.metrics / "video_model_grid.csv", index=False)
    top = G.auc_oof.max()
    cand = G[G.auc_oof >= top - 0.005].sort_values(["n_features", "grid_idx"])     # a parita' (entro 0,005 di AUC) il piu' semplice
    best = cand.iloc[0]
    fname, aggs, params_v = best.feature_set, tuple(best.aggs.split("+")), json.loads(best.params)
    make_v = partial(make_model, spec, params_v)
    cols = fsets[fname]
    log.info("modello di video scelto in CV: feature_set=%s aggs=%s params=%s (AUC OOF %.3f; migliore assoluto %.3f)",
             fname, aggs, params_v, best.auc_oof, top)
    Fv_all = video_matrix(X_all, tr_ids + te_ids, aggs, cols)
    Fv_tr, Fv_te = Fv_all.loc[tr_ids], Fv_all.loc[te_ids]
    p_oof = cvm.oof_video(Fv_tr, y_tr, folds, make_v, n_jobs)

    # ---------------- 2. modello a finestre
    Xw, Twt = X_all[cols], Tw
    if is_fill is not None:
        Xw = Xw.assign(is_fill=is_fill.to_numpy())
    meta_v = info.loc[tr_ids + te_ids, ["is_pos", "g", "dur"]].copy()
    wm = cfg["window_model"]
    deltas = wm["deltas"]
    d_ref = 2 if 2 in deltas else deltas[len(deltas) // 2]
    res = cvm.oof_window_many(Xw, Twt, meta_v, tr_ids, folds, [(partial(make_model, spec, p, "window"), d_ref) for p in spec["window_grid"]],
                              time_col, n_jobs)
    cw = [roc_auc_score(yw, s) if len(set(yw)) == 2 else 0.0 for s, yw in res]
    wi = int(np.argmax(cw))                                                     # a parita' il primo (il piu' semplice)
    params_w = spec["window_grid"][wi]
    make_w = partial(make_model, spec, params_w, "window")
    log.info("modello a finestre: params=%s (AUC finestre OOF %.3f)", params_w, cw[wi])
    best_t = {}
    for d, (s, yw) in zip(deltas, cvm.oof_window_many(Xw, Twt, meta_v, tr_ids, folds, [(make_w, d) for d in deltas], time_col, n_jobs)):
        bw = cvm.best_windows(s, Twt, time_col)
        best_t[d] = np.array([bw[v][1] if v in bw else np.nan for v in tr_ids])
    dur_tr = meta_v.loc[tr_ids, "dur"].to_numpy(float)
    g_arr = meta_v.loc[tr_ids, "g"].to_numpy(float)
    pos_arr = y_tr.astype(bool)
    has_win = np.array([not np.isnan(best_t[deltas[0]][i]) for i in range(len(tr_ids))])
    p_for_thr = np.where(has_win, p_oof, -1.0)
    f1_cv, delta, thr, f1_by_delta = grid_search(p_for_thr, best_t, dur_tr, pos_arr, g_arr, deltas)
    log.info("scelti in CV: delta=%+d s, soglia=%.3f -> F1 challenge OOF %.3f", delta, thr, f1_cv)

    # esiti OOF del train
    rows_tr = []
    for i, v in enumerate(tr):
        ans = None
        if has_win[i] and p_oof[i] >= thr:
            ans = int(answer_int(best_t[delta][i], delta, dur_tr[i]))
        rows_tr.append(dict(video=v.vid, split=v.split, event_type=v.event_type, is_pos=v.is_pos, g=v.g, dur=dur_tr[i],
                            p_video=p_oof[i], has_windows=bool(has_win[i]), answer=ans, outcome=outcome(v.is_pos, v.g, ans)))
    P_tr = pd.DataFrame(rows_tr)
    cv_c = from_outcomes(P_tr.outcome)
    cv_res = dict(auc_detect=float(roc_auc_score(y_tr, p_oof)), f1_detect=_f1_detect(y_tr, p_oof, thr),
                  f1_challenge=cv_c["f1"], precision=cv_c["precision"], recall=cv_c["recall"],
                  TP=cv_c["TP"], FP=cv_c["FP"], FN=cv_c["FN"], TN=cv_c["TN"], f1_by_delta={str(k): float(v) for k, v in f1_by_delta.items()})

    # ---------------- 3. modelli finali sul train
    video_model = make_v().fit(Fv_tr, y_tr)
    vid_w = Xw.index.get_level_values("video").to_numpy()
    m_trw = np.isin(vid_w, tr_ids)
    y_w = cvm.window_labels(Twt[m_trw], meta_v, delta, time_col)
    window_model = make_w().fit(Xw[m_trw], y_w)

    # ---------------- 4. test
    test_res, P_te, W_te = None, pd.DataFrame(), pd.DataFrame()
    if te:
        y_te = np.array([v.is_pos for v in te]).astype(int)
        p_te = video_model.predict_proba(Fv_te)
        m_tew = np.isin(vid_w, te_ids)
        sc_te = pd.Series(window_model.predict_proba(Xw[m_tew]), index=Xw.index[m_tew]) if m_tew.any() else pd.Series(dtype=float)
        bw_te = cvm.best_windows(sc_te, Twt, time_col) if len(sc_te) else {}
        rows = []
        for i, v in enumerate(te):
            dur_v = float(meta_v.loc[v.vid, "dur"])
            ans, best_w, best_s = None, None, None
            if v.vid in bw_te:
                best_s, t_b, best_w = bw_te[v.vid]
                if p_te[i] >= thr:
                    ans = int(answer_int(t_b, delta, dur_v))
            rows.append(dict(video=v.vid, split=v.split, event_type=v.event_type, is_pos=v.is_pos, g=v.g, dur=dur_v, p_video=p_te[i],
                             has_windows=v.vid in bw_te, best_grid=best_w, best_grid_score=best_s, answer=ans,
                             outcome=outcome(v.is_pos, v.g, ans)))
        P_te = pd.DataFrame(rows)
        c = from_outcomes(P_te.outcome)
        ci = bootstrap(P_te.p_video.to_numpy(), thr, P_te.is_pos.to_numpy(), P_te.outcome.to_numpy(), cfg["bootstrap"])
        per_type = {}
        for t in ("static dumping", "dynamic dumping"):
            sub = P_te[P_te.event_type == t]
            per_type[t] = dict(n=len(sub), TP=int((sub.outcome == "TP").sum()), FN=int((sub.outcome == "FN").sum()),
                               wrong_time=int((sub.outcome == "FP").sum()))
        neg = P_te[~P_te.is_pos]
        per_type["normal"] = dict(n=len(neg), false_alarms=int((neg.outcome == "FP").sum()))
        test_res = dict(auc_detect=float(roc_auc_score(y_te, p_te)), f1_detect=_f1_detect(y_te, p_te, thr),
                        f1_challenge=c["f1"], precision=c["precision"], recall=c["recall"], TP=c["TP"], FP=c["FP"], FN=c["FN"], TN=c["TN"],
                        ci95=ci, per_type=per_type, n=len(te), n_pos=int(y_te.sum()))
        log.info("TEST: F1 challenge %.3f [%.3f, %.3f] (P %.2f R %.2f; TP %d FP %d FN %d TN %d) | rilevamento AUC %.3f F1 %.3f",
                 c["f1"], ci["f1_challenge"][0], ci["f1_challenge"][1], c["precision"], c["recall"], c["TP"], c["FP"], c["FN"], c["TN"],
                 test_res["auc_detect"], test_res["f1_detect"])
        if len(sc_te):
            W_te = sc_te.rename("score").reset_index().merge(Twt.reset_index(), on=["video", "win"])
            W_te["valid"] = cvm.window_labels(Twt.loc[sc_te.index], meta_v, delta, time_col)

    # ---------------- 5. baseline di sola scorciatoia (metadati) con lo stesso protocollo
    base_res = None
    M = _shortcut_features(info, meta, tr_ids + te_ids)
    m_tr = [v for v in tr_ids if v in M.index]
    if len(m_tr) > 20:
        yb = info.loc[m_tr, "is_pos"].to_numpy(int)
        fb = cvm.make_folds(yb, np.array([gid[groups.get(v, f"solo_{v}")] for v in m_tr]), cfg["cv"]["n_splits"], cfg["cv"]["repeats"], cfg["cv"]["seed"])
        oofs = cvm.oof_video_many([(M.loc[m_tr], partial(make_model, spec, p)) for p in spec["video_grid"]], yb, fb, n_jobs)
        aucs = [roc_auc_score(yb, o) for o in oofs]
        bi = int(np.argmax(aucs))
        oof_b, params_b = oofs[bi], spec["video_grid"][bi]
        thr_b = _best_thr_detect(yb, oof_b)
        base_res = dict(cv_auc=float(aucs[bi]), cv_f1_detect=_f1_detect(yb, oof_b, thr_b), params=params_b, C=params_b.get("C"))
        m_te = [v for v in te_ids if v in M.index]
        if m_te:
            mod = make_model(spec, params_b).fit(M.loc[m_tr], yb)
            pb = mod.predict_proba(M.loc[m_te])
            ybt = info.loc[m_te, "is_pos"].to_numpy(int)
            base_res.update(test_auc=float(roc_auc_score(ybt, pb)), test_f1_detect=_f1_detect(ybt, pb, thr_b))

    # ---------------- 6. statistiche per domanda, errori, probabilita', modelli
    stats = _question_stats(X_all, info, tr_ids, te_ids, cols)
    stats.round(4).to_csv(run.tables / "question_stats.csv")
    P_tr.round(4).to_csv(run.tables / "predictions_train_oof.csv", index=False)
    err_tr = P_tr[P_tr.outcome.isin(["FP", "FN"])].assign(error_type=lambda d: np.where(
        d.outcome == "FN", "non rilevato", np.where(d.is_pos, "tempo sbagliato", "falso allarme")))
    err_tr.round(4).to_csv(run.tables / "errors_train_oof.csv", index=False)
    if len(P_te):
        P_te.round(4).to_csv(run.tables / "predictions_test.csv", index=False)
        err_te = P_te[P_te.outcome.isin(["FP", "FN"])].assign(error_type=lambda d: np.where(
            d.outcome == "FN", "non rilevato", np.where(d.is_pos, "tempo sbagliato", "falso allarme")))
        err_te.round(4).to_csv(run.tables / "errors_test.csv", index=False)
        W_te.round(4).to_csv(run.tables / "window_scores_test.csv", index=False)
    export_probabilities(run, long, questions, info, tr_ids + te_ids, pd.concat([P_tr, P_te]) if len(P_te) else P_tr)

    md = run.models / model_tag
    md.mkdir(parents=True, exist_ok=True)
    joblib.dump(video_model, md / "video_model.joblib")
    joblib.dump(window_model, md / "window_model.joblib")
    final = dict(run=run.name, classifier=spec["name"], kind=spec["kind"], video_kind=spec.get("video_kind", spec["kind"]),
                 window_kind=spec.get("window_kind", spec["kind"]), video_params=params_v, window_params=params_w,
                 trained_on=sorted({v.split for v in tr}), n_train_videos=len(tr), time_col=time_col, delta=int(delta),
                 threshold=float(thr), feature_set=fname, aggs=list(aggs), question_ids=[q.id for q in questions],
                 questions_hash=json.loads((run.features / "_manifest.json").read_text())["questions_hash"],
                 context_mode=json.loads((run.features / "_manifest.json").read_text())["context_mode"],
                 max_pixels=json.loads((run.features / "_manifest.json").read_text())["max_pixels"],
                 clip_logit=clip, clip_derived=clip_d, derived=derived,
                 video_model=video_model.to_dict(), window_model=window_model.to_dict())
    (md / "final_config.json").write_text(json.dumps(final, indent=1), encoding="utf-8")

    summary = dict(run=run.name, classifier=spec["name"], kind=spec["kind"], video_kind=spec.get("video_kind", spec["kind"]),
                   window_kind=spec.get("window_kind", spec["kind"]), time_col=time_col, n_questions=len(questions),
                   train=dict(n_videos=len(tr), n_pos=int(y_tr.sum()), skipped_no_features=len(skipped), n_grids=int(m_trw.sum()),
                              n_fill_grids=int(is_fill[m_trw].sum()) if is_fill is not None else 0),
                   selection=dict(feature_set=fname, aggs=list(aggs), video_params=params_v, window_params=params_w,
                                  C_video=params_v.get("C"), C_window=params_w.get("C"), delta=int(delta), threshold=float(thr),
                                  n_video_features=int(Fv_tr.shape[1]), n_window_features=int(Xw.shape[1])),
                   cv_train=cv_res, test=test_res, baseline_shortcuts=base_res)
    (run.metrics / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def export_probabilities(run, long, questions, info, vids, preds):
    """Una riga per griglia con tutte le probabilita' di Visual-JEV (softmax dei logit) e l'esito del video."""
    Pm = window_probabilities(long, questions).reset_index()
    wins = pd.concat([pd.read_parquet(run.features / f"{v}.windows.parquet") for v in vids
                      if (run.features / f"{v}.windows.parquet").exists()], ignore_index=True)
    keep = ["video", "win", "image", "t_start", "t_end"] + (["kind"] if "kind" in wins.columns else [])
    out = wins[keep].merge(Pm, on=["video", "win"], how="inner")
    meta = info[["split", "is_pos", "g", "event_type"]].reset_index().rename(columns={"index": "video"})
    out = meta.merge(out, on="video", how="inner")
    if len(preds):
        out = out.merge(preds[["video", "p_video", "answer", "outcome"]], on="video", how="left")
    out.to_parquet(run.tables / "probabilities_grids.parquet", index=False)
    out.to_csv(run.tables / "probabilities_grids.csv", index=False, encoding="utf-8-sig")
