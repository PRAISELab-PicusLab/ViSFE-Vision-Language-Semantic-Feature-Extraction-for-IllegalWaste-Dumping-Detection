"""Dalla cache dei logit alle matrici di feature: per griglia (finestra) e per video.

Feature per griglia ("segnali"):
  - domande sì/no:        logit di "sì" = l0 - l1            -> colonna <id>
  - affermazioni/scelta:  log-softmax delle opzioni indicate in feature_options -> colonne <id>#k
  - derivate (opzionali): dep_state / pick_state, combinazioni delle quattro domande di direzione
Per video: massimo, media, media dei 3 migliori e frazione di griglie sopra 0,5, per ogni segnale, piu' l'indicatore 'no_events'.
NESSUNA scorciatoia (durata, risoluzione, fps, luminosita', notte, numero di griglie) entra fra le feature.
"""
import numpy as np
import pandas as pd

WIN_COLS = ["video", "win", "t_start", "t_center", "t_end", "image", "image_tokens", "t_win", "kind"]
LONG_COLS = ["video", "win", "qid", "n_opt", "l0", "l1", "l2", "l3"]


def read_features(run, vids):
    """(long, wins, have): `have` = video con le feature complete (anche se senza griglie)."""
    longs, wins, have = [], [], set()
    for v in vids:
        p, w = run.features / f"{v}.parquet", run.features / f"{v}.windows.parquet"
        if not (p.exists() and w.exists()):
            continue
        have.add(v)
        d, ww = pd.read_parquet(p), pd.read_parquet(w)
        if len(d):
            longs.append(d)
        if len(ww):
            wins.append(ww)
    long = pd.concat(longs, ignore_index=True) if longs else pd.DataFrame({c: [] for c in LONG_COLS})
    win = pd.concat(wins, ignore_index=True) if wins else pd.DataFrame({c: [] for c in WIN_COLS})
    return long, win, have


def _log_softmax(L):
    L = L - L.max(axis=1, keepdims=True)
    return L - np.log(np.exp(L).sum(axis=1, keepdims=True))


def window_matrix(long, questions, derived=None, clip=8.0, clip_derived=16.0):
    """DataFrame indicizzato (video, win): una colonna per segnale. Le derivate vanno in coda."""
    idx = long[["video", "win"]].drop_duplicates().sort_values(["video", "win"])
    midx = pd.MultiIndex.from_frame(idx)
    cols = {}
    for q in questions:
        sub = long[long["qid"] == q.id].set_index(["video", "win"]).sort_index()
        if sub.empty:
            continue
        L = sub[["l0", "l1", "l2", "l3"]].to_numpy(dtype=float)[:, :q.n_options]
        if q.kind == "yesno":
            cols[q.id] = pd.Series(L[:, 0] - L[:, 1], index=sub.index)
        else:
            lsm = _log_softmax(L)
            for k in q.feature_options:
                cols[f"{q.id}#{k}"] = pd.Series(lsm[:, k], index=sub.index)
    X = pd.DataFrame(cols).reindex(midx).astype("float64")
    X = X.clip(-clip, clip)
    for name, spec in (derived or {}).items():
        if all(c in X.columns for c in spec["plus"] + spec["minus"]):
            X[name] = (sum(X[c] for c in spec["plus"]) - sum(X[c] for c in spec["minus"])).clip(-clip_derived, clip_derived)
    return X


def base_columns(X, derived):
    return [c for c in X.columns if c not in (derived or {})]


def window_probabilities(long, questions):
    """Probabilita' (softmax di tutti i logit) per griglia: una colonna per domanda (sì/no) o per opzione. Per l'esportazione."""
    idx = long[["video", "win"]].drop_duplicates().sort_values(["video", "win"])
    midx = pd.MultiIndex.from_frame(idx)
    cols = {}
    names_it = {"supported": "supportata", "contradicted": "contraddetta", "not determined": "non_determinata"}
    for q in questions:
        sub = long[long["qid"] == q.id].set_index(["video", "win"]).sort_index()
        if sub.empty:
            continue
        L = sub[["l0", "l1", "l2", "l3"]].to_numpy(dtype=float)[:, :q.n_options]
        P = np.exp(_log_softmax(L))
        if q.kind == "yesno":
            cols[q.id] = pd.Series(P[:, 0], index=sub.index)
        elif q.kind == "claim":
            for k, o in enumerate(q.options):
                cols[f"{q.id}__{names_it[o]}"] = pd.Series(P[:, k], index=sub.index)
        else:
            for k in range(q.n_options):
                cols[f"{q.id}__opz{k + 1}"] = pd.Series(P[:, k], index=sub.index)
    return pd.DataFrame(cols).reindex(midx).round(4)


def video_matrix(X, vids, aggs, cols=None):
    """Una riga per video: <segnale>|<agg> + no_events (1 se il video non ha griglie)."""
    cols = list(cols if cols is not None else X.columns)
    thr = np.array([np.log(0.5) if "#" in c else 0.0 for c in cols])
    by_video = {v: g[cols].to_numpy() for v, g in X.groupby(level="video")} if len(X) else {}
    names = [f"{c}|{a}" for c in cols for a in aggs]
    rows = {}
    for v in vids:
        if v in by_video:
            A = by_video[v]
            S = np.sort(A, axis=0)[::-1]
            parts = {"max": S[0], "mean": A.mean(axis=0), "top3": S[:3].mean(axis=0), "frac": (A > thr).mean(axis=0)}
            row = [parts[a][i] for i, _ in enumerate(cols) for a in aggs]
            rows[v] = row + [0.0]
        else:
            rows[v] = [np.nan] * len(names) + [1.0]
    return pd.DataFrame.from_dict(rows, orient="index", columns=names + ["no_events"])
