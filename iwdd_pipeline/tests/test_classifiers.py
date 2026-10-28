"""Classificatori alternativi, parti dei job array e confronto (nessuna GPU). Scrive solo in cartelle temporanee."""
import argparse
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))         # per test_basic.synth_long
os.environ.setdefault("IWDD_RUNS", tempfile.mkdtemp(prefix="iwdd_clf_runs_"))
os.environ["IWDD_CLASSIFIER_OUT"] = tempfile.mkdtemp(prefix="iwdd_clf_out_")     # mai in classifiers/<nome>/runs/

from iwdd import cli                                                       # noqa: E402
from iwdd.config import load_config                                        # noqa: E402
from iwdd.data import Video, get_videos                                    # noqa: E402
from iwdd.models import KINDS, list_classifiers, load_spec, make_model    # noqa: E402
from iwdd.paths import Run                                                 # noqa: E402
from iwdd.questions import load_derived, load_questions, questions_hash   # noqa: E402
from test_basic import synth_long                                          # noqa: E402


def synthetic_run(name):
    """Un run con feature finte (stessa costruzione dell'end-to-end di test_basic): 60 video di train, 24 di test."""
    qs = load_questions()
    run = Run(name)
    rng, pyr = np.random.default_rng(1), random.Random(1)
    vids, groups = [], {}
    for split, n in (("train", 60), ("test", 24)):
        for i in range(n):
            pos = i % 2 == 0
            vid = f"{'t' if split == 'train' else 'x'}{i:03d}"
            g = float(pyr.randint(2, 20)) if pos else -1.0
            vids.append(Video(vid, split, pos, g, ("static dumping" if i % 4 == 0 else "dynamic dumping") if pos else "normal",
                              30.0, "day", "1280x720"))
            n_win = pyr.randint(2, 8)
            synth_long(qs, vid, n_win, pos, rng).to_parquet(run.features / f"{vid}.parquet", index=False)
            ts = [max(0.0, g - 2) if pos else 0.0] + [4.0 * (k + 6) for k in range(n_win - 1)]
            pd.DataFrame({"video": vid, "win": range(n_win), "t_start": ts, "t_center": [t + 1.9 for t in ts],
                          "t_end": [t + 3.75 for t in ts], "image": [f"{vid}_e{k}.jpg" for k in range(n_win)],
                          "image_tokens": 900, "t_win": 1.0}).to_parquet(run.features / f"{vid}.windows.parquet", index=False)
            (run.meta / f"{vid}.json").write_text(json.dumps(dict(width=1280, height=720, t_last=30.0, fps_mean=25.0,
                                                                  brightness=100.0, bg_signature=list(rng.normal(size=576)))))
            groups[vid] = len(groups)
    (run.meta / "groups.json").write_text(json.dumps({"sim": 0.95, "group": groups}))
    (run.features / "_manifest.json").write_text(json.dumps(
        {"questions_hash": questions_hash(qs, "rows", 1003520), "context_mode": "rows", "max_pixels": 1003520,
         "question_ids": [q.id for q in qs]}))
    return vids


class TestSpecifiche(unittest.TestCase):
    def test_cartelle_e_specifiche(self):
        names = list_classifiers()
        for n in ("logreg", "logreg_enet", "svm", "decision_tree", "random_forest", "hist_gb", "mlp"):
            self.assertIn(n, names)
        for n in names:
            s = load_spec(n)
            self.assertIn(s["kind"], KINDS)
            self.assertTrue(s["video_grid"] and s["window_grid"], n)

    def test_ogni_parametro_addestra(self):
        """Ogni voce di ogni griglia costruisce un modello che addestra e da' punteggi in [0, 1], anche con NaN nelle feature."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("abcdef"))
        X.iloc[::7, 2] = np.nan
        y = (X["a"].fillna(0) + 0.3 * rng.normal(size=120) > 0).astype(int).to_numpy()
        for n in list_classifiers():
            s = load_spec(n)
            for params in s["video_grid"] + s["window_grid"]:
                p = make_model(s, params).fit(X, y).predict_proba(X)
                self.assertEqual(p.shape, (120,), (n, params))
                self.assertTrue(np.all((p >= 0) & (p <= 1)), (n, params))


class TestParti(unittest.TestCase):
    def test_parti_disgiunte_e_complete(self):
        """Le parti del job array coprono tutti i video una volta sola, sempre uguali (scan/prepare/extract vedono la stessa parte)."""
        ldir = Path(tempfile.mkdtemp(prefix="iwdd_ds_"))
        for split, n in (("train", 37), ("test", 11)):
            (ldir / split / "labels").mkdir(parents=True)
            for i in range(n):
                (ldir / split / "labels" / f"{split[:2]}{i:03d}.json").write_text(json.dumps(
                    {"label": "false", "timestamp": -1, "metadata": {"event_type": "normal", "duration": 5.0 + (i * 7) % 40}}))
        import iwdd.data as data
        old = data.DATASET
        data.DATASET = ldir
        try:
            allv = {v.vid for v in get_videos(["train", "test"])}
            seen = []
            for k in range(6):
                a = argparse.Namespace(splits="train,test", limit_train=None, limit_test=None, part=k, num_parts=6)
                part = [v.vid for v in cli._videos(a)]
                self.assertEqual(part, [v.vid for v in cli._videos(a)])       # deterministica
                seen += part
            self.assertEqual(sorted(seen), sorted(allv))
            self.assertEqual(len(seen), len(set(seen)))
        finally:
            data.DATASET = old


class TestEndToEndClassificatori(unittest.TestCase):
    """train-eval con ogni classificatore (griglie ridotte, CV in parallelo) e confronto finale."""

    def test_tutti(self):
        from iwdd.report import write_report
        from iwdd.train_eval import run_train_eval
        vids = synthetic_run("sint_clf")
        cfg = load_config()
        cfg["cv"]["repeats"] = 1
        cfg["bootstrap"] = 30
        cfg["video_model"]["aggs_options"] = [["max"], ["max", "mean"]]
        cfg["window_model"]["deltas"] = [0, 2]
        qs, derived = load_questions(), load_derived()
        tr = [v for v in vids if v.split == "train"]
        te = [v for v in vids if v.split == "test"]
        for n in list_classifiers():
            spec = load_spec(n)
            # griglie ridotte alle voci meno regolarizzate (le prime possono dare un modello costante, es. L1 forte)
            spec["video_grid"], spec["window_grid"] = spec["video_grid"][-2:], spec["window_grid"][-1:]
            run = Run("sint_clf", classifier=n)
            s = run_train_eval(run, cfg, qs, derived, tr, te, spec=spec, n_jobs=2)
            self.assertEqual(s["classifier"], n)
            self.assertGreater(s["test"]["auc_detect"], 0.7, n)
            self.assertTrue(str(run.out).startswith(os.environ["IWDD_CLASSIFIER_OUT"]), run.out)
            for rel in ("metrics/summary.json", "tables/predictions_test.csv", "models/default/final_config.json",
                        "models/default/video_model.joblib", "models/default/window_model.joblib"):
                self.assertTrue((run.out / rel).exists(), (n, rel))
            self.assertFalse((run.root / "metrics" / "summary.json").exists())    # il run condiviso non viene sporcato
            self.assertTrue(write_report(run, qs).exists())
        self.assertEqual(cli.main(["compare", "--run", "sint_clf"]), 0)
        table = pd.read_csv(Path(os.environ["IWDD_CLASSIFIER_OUT"]) / "comparison" / "sint_clf.csv")
        self.assertEqual(sorted(table["classificatore"]), list_classifiers())


class TestCoppie(unittest.TestCase):
    """Coppie A+B: si'/no della specifica A, tempo della specifica B."""

    def test_specifica_composta(self):
        s = load_spec("mlp+random_forest")
        self.assertEqual((s["video_kind"], s["window_kind"]), ("mlp", "random_forest"))
        self.assertEqual(s["video_grid"], load_spec("mlp")["video_grid"])
        self.assertEqual(s["window_grid"], load_spec("random_forest")["window_grid"])
        self.assertEqual(type(make_model(s, s["video_grid"][0])).__name__, "Clf")
        self.assertEqual(make_model(s, s["video_grid"][0]).kind, "mlp")
        self.assertEqual(make_model(s, s["window_grid"][0], "window").kind, "random_forest")

    def test_diagonale_identica_e_uscite_separate(self):
        from iwdd.train_eval import run_train_eval
        vids = synthetic_run("sint_pair")
        cfg = load_config()
        cfg["cv"]["repeats"] = 1
        cfg["bootstrap"] = 20
        cfg["video_model"]["aggs_options"] = [["max"], ["max", "mean"]]
        cfg["window_model"]["deltas"] = [0, 2]
        qs, derived = load_questions(), load_derived()
        tr = [v for v in vids if v.split == "train"]
        te = [v for v in vids if v.split == "test"]
        res = {}
        for name in ("logreg", "logreg+logreg", "logreg+decision_tree"):
            spec = load_spec(name)
            spec["video_grid"], spec["window_grid"] = spec["video_grid"][-2:], spec["window_grid"][-2:]
            run = Run("sint_pair", classifier=name)
            res[name] = run_train_eval(run, cfg, qs, derived, tr, te, spec=spec, n_jobs=2)
            self.assertTrue((run.out / "metrics" / "summary.json").exists())
        for key in ("cv_train", "test", "selection"):                          # A+A riproduce A esattamente
            a, b = dict(res["logreg"][key]), dict(res["logreg+logreg"][key])
            self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True), key)
        out = Path(os.environ["IWDD_CLASSIFIER_OUT"])
        self.assertTrue((out / "combinations" / "logreg+decision_tree" / "sint_pair" / "metrics" / "summary.json").exists())
        self.assertEqual(res["logreg+decision_tree"]["window_kind"], "decision_tree")
        self.assertEqual(cli.main(["compare", "--run", "sint_pair", "--combinations"]), 0)
        md = (out / "comparison" / "sint_pair_combinations.md").read_text()
        self.assertIn("logreg+decision_tree", md)


if __name__ == "__main__":
    unittest.main()
