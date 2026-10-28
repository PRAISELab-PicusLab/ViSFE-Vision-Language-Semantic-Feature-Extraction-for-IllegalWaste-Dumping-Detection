"""Classificatori usati sia per il modello di video sia per il modello a finestre.

Tutti condividono lo stesso trattamento dei dati: imputazione con la media del train, standardizzazione sul solo train e classi
bilanciate (class_weight, oppure pesi per campione dove lo stimatore non ha class_weight). predict_proba restituisce sempre un
punteggio in [0, 1] (per l'SVM: sigmoide della funzione di decisione), perche' soglia e scelta della griglia migliore lo richiedono.

- LR: la regressione logistica L2 originale (parametro C). E' il default quando non si indica un classificatore.
- Clf: un classificatore qualsiasi descritto da (kind, params); i kind disponibili sono in KINDS. Le specifiche (griglie di
  iperparametri) stanno in classifiers/<nome>/model.json.
- Coppie "A+B": classificatore di video (si'/no) della specifica A e modello a finestre (il "quando") della specifica B, ognuno con
  la sua griglia. "A+A" equivale ad A. Le uscite vanno in classifiers/combinations/A+B/ (vedi paths.classifier_out).
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .paths import CLASSIFIERS_DIR


class LR:
    def __init__(self, C=0.1):
        self.C = C

    def fit(self, X, y):
        self.cols = list(X.columns)
        self.mu = X.mean().fillna(0.0)
        Xf = X.fillna(self.mu).fillna(0.0)
        self.scaler = StandardScaler().fit(Xf)
        self.lr = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000).fit(self.scaler.transform(Xf), y)
        return self

    def predict_proba(self, X):
        Xf = X[self.cols].fillna(self.mu).fillna(0.0)
        return self.lr.predict_proba(self.scaler.transform(Xf))[:, 1]

    def to_dict(self):
        return {"C": self.C, "features": self.cols, "impute_mean": [float(x) for x in self.mu],
                "scale_mean": [float(x) for x in self.scaler.mean_], "scale_std": [float(x) for x in self.scaler.scale_],
                "coef": [float(x) for x in self.lr.coef_[0]], "intercept": float(self.lr.intercept_[0])}

    @staticmethod
    def from_dict(d):
        m = LR(d["C"])
        m.cols = d["features"]
        m.mu = pd.Series(d["impute_mean"], index=m.cols)
        m.scaler = StandardScaler()
        m.scaler.mean_ = np.array(d["scale_mean"])
        m.scaler.scale_ = np.array(d["scale_std"])
        m.scaler.var_ = m.scaler.scale_ ** 2
        m.scaler.n_features_in_ = len(m.cols)
        m.lr = LogisticRegression()
        m.lr.coef_ = np.array([d["coef"]])
        m.lr.intercept_ = np.array([d["intercept"]])
        m.lr.classes_ = np.array([0, 1])
        return m


# ---------------------------------------------------------------------------------------------------------------------- generici
def _logreg(p):
    l1 = float(p.get("l1_ratio", 0.0))
    return LogisticRegression(C=p["C"], l1_ratio=l1, solver="saga" if l1 > 0 else "lbfgs", class_weight="balanced",
                              max_iter=p.get("max_iter", 5000), random_state=0)


def _svm(p):
    from sklearn.svm import SVC
    return SVC(C=p["C"], kernel=p.get("kernel", "rbf"), gamma=p.get("gamma", "scale"), class_weight="balanced",
               cache_size=1000, random_state=0)


def _tree(p):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=p.get("max_depth"), min_samples_leaf=p.get("min_samples_leaf", 1),
                                  class_weight="balanced", random_state=0)


def _forest(p):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=p.get("n_estimators", 500), max_depth=p.get("max_depth"),
                                  max_features=p.get("max_features", "sqrt"), min_samples_leaf=p.get("min_samples_leaf", 1),
                                  class_weight="balanced_subsample", n_jobs=1, random_state=0)


def _hgb(p):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(learning_rate=p.get("learning_rate", 0.05), max_iter=p.get("max_iter", 200),
                                          max_depth=p.get("max_depth"), max_leaf_nodes=p.get("max_leaf_nodes", 15),
                                          min_samples_leaf=p.get("min_samples_leaf", 20),
                                          l2_regularization=p.get("l2_regularization", 0.0), class_weight="balanced",
                                          early_stopping=False, random_state=0)


def _mlp(p):
    from sklearn.neural_network import MLPClassifier
    # lbfgs per pochi campioni (classificatore di video, ~400), adam per molti (modello a finestre, migliaia di griglie)
    return MLPClassifier(hidden_layer_sizes=tuple(p.get("hidden", [32])), alpha=p.get("alpha", 1e-2), solver=p.get("solver", "adam"),
                         learning_rate_init=p.get("learning_rate_init", 1e-3), max_iter=p.get("max_iter", 1000),
                         random_state=0)


# kind -> (costruttore, lo stimatore accetta class_weight?)
KINDS = {"logreg": (_logreg, True), "svm": (_svm, True), "decision_tree": (_tree, True), "random_forest": (_forest, True),
         "hist_gb": (_hgb, True), "mlp": (_mlp, False)}


class Clf:
    """Stesso protocollo di LR (imputazione + standardizzazione + classi bilanciate) per qualunque kind di KINDS."""

    def __init__(self, kind, params):
        if kind not in KINDS:
            raise ValueError(f"kind sconosciuto: {kind} (disponibili: {', '.join(KINDS)})")
        self.kind, self.params = kind, dict(params)

    def fit(self, X, y):
        self.cols = list(X.columns)
        self.mu = X.mean().fillna(0.0)
        Xf = X.fillna(self.mu).fillna(0.0)
        self.scaler = StandardScaler().fit(Xf)
        build, has_cw = KINDS[self.kind]
        self.est = build(self.params)
        y = np.asarray(y).astype(int)
        if has_cw:
            self.est.fit(self.scaler.transform(Xf), y)
        else:
            self.est.fit(self.scaler.transform(Xf), y, sample_weight=compute_sample_weight("balanced", y))
        return self

    def predict_proba(self, X):
        Z = self.scaler.transform(X[self.cols].fillna(self.mu).fillna(0.0))
        if self.kind != "svm":          # SVC senza probability=True: niente Platt scaling (lento e con una CV interna)
            return self.est.predict_proba(Z)[:, 1]
        return 1.0 / (1.0 + np.exp(-self.est.decision_function(Z)))

    def to_dict(self):
        d = {"kind": self.kind, "params": self.params, "features": self.cols}
        if self.kind == "logreg":
            d.update(coef=[float(x) for x in self.est.coef_[0]], intercept=float(self.est.intercept_[0]))
        return d


# ---------------------------------------------------------------------------------------------------------------------- specifiche
def legacy_spec(cfg):
    """Il comportamento originale (regressione logistica L2, griglie di C in config/default.json) come specifica."""
    return {"name": None, "kind": "lr_legacy",
            "video_grid": [{"C": c} for c in sorted(cfg["video_model"]["C_grid"])],
            "window_grid": [{"C": c} for c in sorted(cfg["window_model"]["C_grid"])]}


def load_spec(name):
    """classifiers/<name>/model.json: kind, video_grid e window_grid (liste di parametri, dal piu' semplice al piu' complesso).
    Con name = "A+B": classificatore di video della specifica A e modello a finestre della specifica B."""
    if "+" in name:
        v_name, w_name = name.split("+", 1)
        v, w = load_spec(v_name), load_spec(w_name)
        vk, wk = v.get("video_kind", v["kind"]), w.get("window_kind", w["kind"])
        return {"name": name, "kind": f"{vk}+{wk}", "video_kind": vk, "window_kind": wk,
                "video_grid": v["video_grid"], "window_grid": w["window_grid"],
                "description": f"si'/no: {v_name} | tempo: {w_name}"}
    path = CLASSIFIERS_DIR / name / "model.json"
    if not path.exists():
        avail = sorted(p.parent.name for p in CLASSIFIERS_DIR.glob("*/model.json"))
        raise FileNotFoundError(f"Classificatore '{name}' non trovato ({path}). Disponibili: {', '.join(avail)}")
    spec = json.loads(path.read_text(encoding="utf-8-sig"))
    spec["name"] = name
    for role in ("video", "window"):
        kind = spec.get(f"{role}_kind", spec["kind"])
        if kind not in KINDS:
            raise ValueError(f"{path}: kind '{kind}' sconosciuto (disponibili: {', '.join(KINDS)})")
    for key in ("video_grid", "window_grid"):
        if not spec.get(key):
            raise ValueError(f"{path}: '{key}' mancante o vuota")
    return spec


def list_classifiers():
    return sorted(p.parent.name for p in CLASSIFIERS_DIR.glob("*/model.json"))


def make_model(spec, params, role="video"):
    """Modello non addestrato per la specifica e i parametri dati. role: 'video' (si'/no) o 'window' (il "quando")."""
    kind = spec.get(f"{role}_kind", spec["kind"])
    if kind == "lr_legacy":
        return LR(params["C"])
    return Clf(kind, params)
