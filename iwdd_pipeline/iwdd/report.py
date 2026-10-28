"""Report leggibile (report.md) a partire da metrics/summary.json e dalle tabelle del run."""
import json
from datetime import datetime

import pandas as pd


def _pct(x):
    return f"{x:.3f}"


def _params(d):
    return ", ".join(f"{k}={v}" for k, v in sorted((d or {}).items())) or "-"


def write_report(run, questions):
    sp = run.metrics / "summary.json"
    if not sp.exists():
        raise FileNotFoundError(f"{sp} non esiste: esegui prima 'train-eval'")
    s = json.loads(sp.read_text(encoding="utf-8"))
    name = s.get("classifier") or "logistica L2 (originale)"
    L = [f"# Report del run `{s['run']}` — classificatore `{name}`", "", f"_Generato il {datetime.now():%Y-%m-%d %H:%M}_", ""]
    mf = run.features / "_manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {}
    sel, tr, cv, te, bl = s["selection"], s["train"], s["cv_train"], s.get("test"), s.get("baseline_shortcuts")

    L += ["## 1. Configurazione", "",
          f"- Domande: **{s['n_questions']}** (hash `{man.get('questions_hash', '?')}`), contesto `{man.get('context_mode', '?')}`, "
          f"budget pixel immagine {man.get('max_pixels', '?')}.",
          f"- Train: **{tr['n_videos']}** video ({tr['n_pos']} positivi), {tr['n_grids']} griglie (di cui {tr.get('n_fill_grids', 0)} di copertura a frame intero); video senza feature saltati: {tr['skipped_no_features']}.",
          f"- Segnali per griglia: {sel['n_window_features']} | feature per video: {sel['n_video_features']}.",
          f"- Scelte fatte in CV a gruppi sul solo train: insieme di feature `{sel['feature_set']}`, aggregazioni `{'+'.join(sel['aggs'])}`, "
          f"classificatore `{name}` (kind `{s.get('kind', 'lr_legacy')}`), parametri video: {_params(sel.get('video_params') or {'C': sel['C_video']})}, "
          f"parametri finestre: {_params(sel.get('window_params') or {'C': sel['C_window']})}, tempo di risposta = `{s['time_col']}` {sel['delta']:+d} s "
          f"(intero, entro il video), soglia di video = {sel['threshold']:.3f}.", ""]

    L += ["## 2. Risultati", "", "| | F1 challenge | P | R | TP / FP / FN / TN | AUC rilevamento | F1 rilevamento |", "|---|---|---|---|---|---|---|"]
    L.append(f"| **CV su train** (out-of-fold) | {_pct(cv['f1_challenge'])} | {cv['precision']:.2f} | {cv['recall']:.2f} | "
             f"{cv['TP']} / {cv['FP']} / {cv['FN']} / {cv['TN']} | {_pct(cv['auc_detect'])} | {_pct(cv['f1_detect'])} |")
    if te:
        ci = te["ci95"]
        L.append(f"| **TEST** ({te['n']} video, {te['n_pos']} positivi) | **{_pct(te['f1_challenge'])}** "
                 f"[{ci['f1_challenge'][0]:.2f}, {ci['f1_challenge'][1]:.2f}] | {te['precision']:.2f} | {te['recall']:.2f} | "
                 f"{te['TP']} / {te['FP']} / {te['FN']} / {te['TN']} | {_pct(te['auc_detect'])} [{ci['auc'][0]:.2f}, {ci['auc'][1]:.2f}] | "
                 f"{_pct(te['f1_detect'])} [{ci['f1_detect'][0]:.2f}, {ci['f1_detect'][1]:.2f}] |")
    L += ["", "F1 della challenge = rilevamento + tempo dentro `[g-3, g+10]` (risposte in secondi interi). "
          "F1 di rilevamento = c'e' uno scarico nel video, senza considerare il tempo. Gli intervalli sono bootstrap al 95% sui video.", ""]
    if te:
        L += ["### Dove si perde sul test", "", "| tipo | positivi | TP | non rilevati | tempo sbagliato |", "|---|---|---|---|---|"]
        for t in ("static dumping", "dynamic dumping"):
            d = te["per_type"][t]
            L.append(f"| {t} | {d['n']} | {d['TP']} | {d['FN']} | {d['wrong_time']} |")
        n = te["per_type"]["normal"]
        L += [f"| negativi | {n['n']} | – | – | falsi allarmi: {n['false_alarms']} |", ""]
    L += ["### F1 della challenge in CV al variare del delta (soglia ottima per ciascuno)", "",
          "| delta (s) | " + " | ".join(cv["f1_by_delta"]) + " |", "|---|" + "---|" * len(cv["f1_by_delta"]),
          "| F1 | " + " | ".join(f"{v:.3f}" for v in cv["f1_by_delta"].values()) + " |", ""]

    L += ["## 3. Baseline di sola scorciatoia", ""]
    if bl:
        L += [f"Stesso classificatore (`{name}`, parametri {_params(bl.get('params') or {'C': bl.get('C')})}) su durata, risoluzione, "
              f"fps, luminosita' e notte (non entrano nel modello vero): "
              f"CV AUC {bl['cv_auc']:.3f}, F1 di rilevamento {bl['cv_f1_detect']:.3f}" +
              (f"; test AUC {bl['test_auc']:.3f}, F1 {bl['test_f1_detect']:.3f}." if "test_auc" in bl else "."),
              "Dire 'positivo' a tutti darebbe F1 di rilevamento ~0,667 con classi bilanciate.", ""]
    else:
        L += ["Non calcolata (metadati di scansione assenti: esegui `scan`).", ""]

    qp = run.tables / "question_stats.csv"
    if qp.exists():
        q = pd.read_csv(qp, index_col=0)
        col = "AUC max train"
        if col in q.columns:
            q = q.assign(_a=(q["AUC max train"] - 0.5).abs()).sort_values("_a", ascending=False).drop(columns="_a")
            L += ["## 4. Correlazione dei segnali con l'etichetta (livello video)", "",
                  "AUC 0,5 = nessuna relazione. Tabella completa: `tables/question_stats.csv`.", "",
                  "| segnale | AUC max (train) | AUC media (train) | r media (train) | AUC max (test) | AUC media (test) |", "|---|---|---|---|---|---|"]
            for k, r in q.head(12).iterrows():
                L.append(f"| `{k}` | {r.get('AUC max train', float('nan')):.2f} | {r.get('AUC mean train', float('nan')):.2f} | "
                         f"{r.get('r mean train', float('nan')):.2f} | {r.get('AUC max test', float('nan')):.2f} | {r.get('AUC mean test', float('nan')):.2f} |")
            L += ["", "Segnali con AUC piu' vicina a 0,5 (poco informativi):", ""]
            for k, r in q.tail(6).iterrows():
                L.append(f"- `{k}`: AUC max train {r.get('AUC max train', float('nan')):.2f}, test {r.get('AUC max test', float('nan')):.2f}")
            L.append("")

    L += ["## 5. File prodotti", "",
          "- `metrics/summary.json` tutti i numeri; `metrics/video_model_grid.csv` la griglia di selezione in CV",
          "- `tables/predictions_test.csv`, `predictions_train_oof.csv`: una riga per video (punteggio, risposta, esito)",
          "- `tables/errors_test.csv`, `errors_train_oof.csv`: video sbagliati con tipo di errore",
          "- `tables/window_scores_test.csv`: punteggio di ogni griglia del test",
          "- `tables/probabilities_grids.csv|parquet`: probabilita' di Visual-JEV per griglia e domanda (tutti i video)",
          "- `models/<tag>/final_config.json` (pesi, soglia, delta, domande) e `*.joblib`", ""]
    run.report.write_text("\n".join(L), encoding="utf-8")
    return run.report
