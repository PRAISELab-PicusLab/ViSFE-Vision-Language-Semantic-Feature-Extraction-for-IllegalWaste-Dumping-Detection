"""Cross-validation a gruppi di camera (i video della stessa telecamera stanno nello stesso fold), ripetuta."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .evaluate import answer_int
from .models import LR


def _factory(model):
    """model: C della logistica L2 (comportamento originale) oppure una funzione senza argomenti che crea un modello nuovo."""
    return model if callable(model) else (lambda: LR(model))


def make_folds(y, groups, n_splits=5, repeats=3, seed=0):
    """[[ (idx_train, idx_test), ...] per ripetizione]. Indici nella lista dei video di addestramento."""
    y = np.asarray(y).astype(int)
    k = max(2, min(n_splits, len(set(groups)), int(np.bincount(y).min())))
    out = []
    for r in range(repeats):
        skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
        out.append(list(skf.split(np.zeros(len(y)), y, groups)))
    return out


def _fit_predict(make, X_tr, y_tr, X_te):
    return make().fit(X_tr, y_tr).predict_proba(X_te)


def _run(tasks, n_jobs):
    """Esegue gli addestramenti (delayed) in parallelo; con n_jobs=1 in sequenza, senza processi."""
    if n_jobs == 1:
        return [f(*a, **k) for f, a, k in tasks]
    from joblib import Parallel
    return Parallel(n_jobs=n_jobs)(tasks)


def oof_video_many(configs, y, folds, n_jobs=1):
    """Probabilita' out-of-fold del classificatore di video per piu' configurazioni [(F, model)], mediate sulle ripetizioni.
    F: DataFrame (una riga per video); model: C oppure costruttore. Tutti gli addestramenti (configurazione x fold) vanno in
    parallelo. Ritorna una lista di array nello stesso ordine di configs."""
    from joblib import delayed
    y = np.asarray(y).astype(int)
    tasks, where = [], []
    for ci, (F, model) in enumerate(configs):
        make = _factory(model)
        for rep in folds:
            for tr, te in rep:
                tasks.append(delayed(_fit_predict)(make, F.iloc[tr], y[tr], F.iloc[te]))
                where.append((ci, te))
    accs = [np.zeros(len(y)) for _ in configs]
    for (ci, te), pred in zip(where, _run(tasks, n_jobs)):
        accs[ci][te] += pred
    return [a / len(folds) for a in accs]


def oof_video(F, y, folds, model, n_jobs=1):
    """Probabilita' out-of-fold del classificatore di video, mediate sulle ripetizioni. F: DataFrame (una riga per video)."""
    return oof_video_many([(F, model)], y, folds, n_jobs)[0]


def window_labels(Tw, meta_v, delta, time_col):
    """1 se rispondere con il tempo della griglia (+delta, intero, entro il video) cade in [g-3, g+10] in un video positivo.
    meta_v: DataFrame indicizzato per video con is_pos, g, dur."""
    v = Tw.index.get_level_values("video")
    pos = meta_v.loc[v, "is_pos"].to_numpy(dtype=bool)
    g = meta_v.loc[v, "g"].to_numpy(dtype=float)
    dur = meta_v.loc[v, "dur"].to_numpy(dtype=float)
    a = answer_int(Tw[time_col].to_numpy(), delta, dur)
    return (pos & (a >= g - 3) & (a <= g + 10)).astype(int)


def oof_window_many(Xw, Tw, meta_v, train_vids, folds, configs, time_col, n_jobs=1):
    """Punteggi out-of-fold del modello a finestre per piu' configurazioni [(model, delta)]: lista di (Serie indicizzata
    (video, win), etichette) nello stesso ordine. Tutti gli addestramenti (configurazione x fold) vanno in parallelo."""
    from joblib import delayed
    vid = Xw.index.get_level_values("video").to_numpy()
    in_train = np.isin(vid, train_vids)
    Xt, Tt = Xw[in_train], Tw[in_train]
    v_t = Xt.index.get_level_values("video").to_numpy()
    tv = np.array(train_vids)
    masks = [(np.isin(v_t, tv[tr]), np.isin(v_t, tv[te])) for rep in folds for tr, te in rep]
    ys = [window_labels(Tt, meta_v, delta, time_col) for _, delta in configs]
    tasks, where = [], []
    for ci, (model, _) in enumerate(configs):
        make, y = _factory(model), ys[ci]
        for m_tr, m_te in masks:
            if y[m_tr].sum() == 0 or y[m_tr].sum() == m_tr.sum():
                continue
            tasks.append(delayed(_fit_predict)(make, Xt[m_tr], y[m_tr], Xt[m_te]))
            where.append((ci, m_te))
    accs = [np.zeros(len(Xt)) for _ in configs]
    for (ci, m_te), pred in zip(where, _run(tasks, n_jobs)):
        accs[ci][m_te] += pred
    return [(pd.Series(a / len(folds), index=Xt.index), y) for a, y in zip(accs, ys)]


def oof_window(Xw, Tw, meta_v, train_vids, folds, model, delta, time_col, n_jobs=1):
    """Punteggi out-of-fold del modello a finestre (Serie indicizzata (video, win)) per un dato delta. model: C oppure costruttore."""
    return oof_window_many(Xw, Tw, meta_v, train_vids, folds, [(model, delta)], time_col, n_jobs)[0]


def best_windows(scores, Tw, time_col):
    """{video: (punteggio massimo, tempo della finestra col punteggio massimo, indice griglia)}."""
    out = {}
    for v, s in scores.groupby(level="video"):
        k = s.idxmax()                       # (video, win)
        out[v] = (float(s.max()), float(Tw.loc[k, time_col]), int(k[1]))
    return out
