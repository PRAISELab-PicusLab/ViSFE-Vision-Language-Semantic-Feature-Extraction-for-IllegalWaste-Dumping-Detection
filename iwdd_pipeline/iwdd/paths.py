"""Percorsi e layout delle cartelle dei risultati. Tutto si configura con variabili d'ambiente.

IWDD_PROJECT  cartella del progetto (contiene main.py, preprocessing.py, decompose_video.py, visual-jev/, dataset/)
              default: la cartella che contiene iwdd_pipeline/
IWDD_DATASET  default: <progetto>/dataset  (con train/{videos,labels} e test/{videos,labels})
IWDD_RUNS     default: <iwdd_pipeline>/runs
IWDD_SCRATCH  cartella temporanea per i frame intermedi del preprocessing (default: $TMPDIR, altrimenti <run>/scratch)

Layout di un run (<IWDD_RUNS>/<nome>/):
  meta/        <video>.json (scansione) + groups.json (gruppi di camera)
  events/      <video>/*.jpg  (immagini evento del preprocessing)
  features/    <video>.parquet (logit grezzi) + <video>.windows.parquet (tempi delle griglie) + _manifest.json + timing_*.jsonl
  models/      final_config.json, *.joblib
  metrics/     summary.json (tutti i numeri)
  tables/      predictions_*.csv, errors_*.csv, question_stats.csv, probabilities_grids.(csv|parquet)
  logs/        errori delle singole fasi
  report.md    riepilogo leggibile
models/, metrics/, tables/ e report.md sono le USCITE dell'addestramento: con un classificatore (train-eval --classifier NOME)
vanno invece in classifiers/<NOME>/runs/<run>/, mentre meta/, events/ e features/ restano condivise nel run.
"""
import os
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = PKG_DIR.parent
CONFIG_DIR = PIPELINE_DIR / "config"
CLASSIFIERS_DIR = PIPELINE_DIR / "classifiers"
PROJECT_ROOT = Path(os.environ.get("IWDD_PROJECT", PIPELINE_DIR.parent)).resolve()
DATASET = Path(os.environ.get("IWDD_DATASET", PROJECT_ROOT / "dataset")).resolve()
RUNS = Path(os.environ.get("IWDD_RUNS", PIPELINE_DIR / "runs")).resolve()


def ensure_project_imports():
    """Rende importabili i moduli del progetto (main.py, preprocessing.py, decompose_video.py) e il codice di Visual-JEV
    (vdm), qualunque sia il layout (visual-jev/code oppure il clone annidato visual-jev/visual-jev/code)."""
    sys.dont_write_bytecode = True      # non lasciare __pycache__ nella cartella del progetto
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    for p in (PROJECT_ROOT / "visual-jev" / "code", PROJECT_ROOT / "visual-jev" / "visual-jev" / "code"):
        if (p / "vdm").is_dir():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return str(p)
    return None


def classifier_out(classifier, run_name):
    """Uscite di un classificatore su un run: classifiers/<classificatore>/runs/<run>/; per una coppia "A+B" (si'/no di A, tempo di B)
    classifiers/combinations/A+B/runs/<run>/. IWDD_CLASSIFIER_OUT sposta tutto (per i test)."""
    base = os.environ.get("IWDD_CLASSIFIER_OUT")
    sub = Path("combinations") / classifier if "+" in classifier else Path(classifier)
    return Path(base) / sub / run_name if base else CLASSIFIERS_DIR / sub / "runs" / run_name


class Run:
    """Cartelle di un run. Le crea solo quando servono (mkdir)."""

    def __init__(self, name, classifier=None):
        self.name = name
        self.root = RUNS / name
        self.classifier = classifier
        # uscite dell'addestramento: nel run (comportamento originale) oppure nella cartella del classificatore
        self.out = classifier_out(classifier, name) if classifier else self.root

    def _d(self, sub, base=None):
        d = (base or self.root) / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def meta(self):
        return self._d("meta")

    @property
    def events(self):
        return self._d("events")

    @property
    def features(self):
        return self._d("features")

    @property
    def models(self):
        return self._d("models", self.out)

    @property
    def metrics(self):
        return self._d("metrics", self.out)

    @property
    def tables(self):
        return self._d("tables", self.out)

    @property
    def logs(self):
        return self._d("logs")

    def scratch(self):
        base = os.environ.get("IWDD_SCRATCH") or os.environ.get("TMPDIR")
        d = (Path(base) / f"iwdd_{self.name}") if base else (self.root / "scratch")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def report(self):
        self.out.mkdir(parents=True, exist_ok=True)
        return self.out / "report.md"

    def __repr__(self):
        return f"Run({self.name!r}, {self.root}, classifier={self.classifier!r})"
