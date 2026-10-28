"""Metrica della challenge: una risposta p e' un TP se g-3 <= p <= g+10 in un video positivo; una risposta in un negativo, o fuori
finestra in un positivo, e' un FP; nessuna risposta in un positivo e' un FN. F1 = 2TP / (2TP + FP + FN).
Stessa regola di main.evaluate() del progetto (verificata in tests/test_evaluate.py). Le risposte sono SECONDI INTERI, come nel
formato di consegna (un file <video>.mp4.txt con un intero)."""
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score


def answer_int(t, delta, dur):
    """Risposta intera: t + delta, non oltre la fine del video, arrotondata al secondo piu' vicino (mai negativa). Accetta array."""
    return np.maximum(0, np.floor(np.minimum(np.asarray(t, dtype=float) + delta, dur) + 0.5))


def outcome(is_pos, g, ans):
    if not is_pos:
        return "FP" if ans is not None else "TN"
    if ans is None:
        return "FN"
    return "TP" if g - 3 <= ans <= g + 10 else "FP"


def prf(c):
    tp, fp, fn, tn = (c.get(k, 0) for k in ("TP", "FP", "FN", "TN"))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r, "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def from_outcomes(outs):
    return prf(Counter(outs))


def grid_search(p, best_t, dur, is_pos, g, deltas):
    """Sceglie (delta, soglia) che massimizzano l'F1 della challenge.
    p: punteggio di video (NaN/-1 = non puo' rispondere); best_t: {delta: tempo della finestra migliore (NaN se nessuna)}.
    Ritorna (f1, delta, soglia, tabella {delta: miglior f1})."""
    p = np.asarray(p, dtype=float)
    can = ~np.isnan(p) & (p >= 0)
    best = (-1.0, None, None)
    per_delta = {}
    for d in deltas:
        t = np.asarray(best_t[d], dtype=float)
        ok = can & ~np.isnan(t)
        ans = np.full(len(p), np.nan)
        ans[ok] = answer_int(t[ok], d, dur[ok])
        hit = is_pos & ok & (ans >= g - 3) & (ans <= g + 10)
        bd = -1.0
        for thr in np.unique(p[ok])[::-1]:
            sel = ok & (p >= thr)
            tp = int((sel & hit).sum())
            fp = int(sel.sum()) - tp
            fn = int((is_pos & ~sel).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            bd = max(bd, f1)
            if f1 > best[0]:
                best = (f1, d, float(thr))
        per_delta[d] = bd
    return best[0], best[1], best[2], per_delta


def bootstrap(p_det, thr, is_pos, outs, n=1000, seed=0):
    """Intervalli al 95% (ricampionando i video): AUC di rilevamento, F1 di rilevamento, F1 della challenge."""
    rng = np.random.default_rng(seed)
    p_det, is_pos = np.asarray(p_det, dtype=float), np.asarray(is_pos)
    outs = np.asarray(outs)
    res = {"auc": [], "f1_detect": [], "f1_challenge": []}
    for _ in range(n):
        i = rng.integers(0, len(p_det), len(p_det))
        if len(set(is_pos[i])) < 2:
            continue
        res["auc"].append(roc_auc_score(is_pos[i], p_det[i]))
        pred = p_det[i] >= thr
        tp, fp, fn = (pred & is_pos[i]).sum(), (pred & ~is_pos[i]).sum(), (~pred & is_pos[i]).sum()
        res["f1_detect"].append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        res["f1_challenge"].append(prf(Counter(outs[i]))["f1"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}
