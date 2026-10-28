"""Test senza GPU. Esecuzione:  cd iwdd_pipeline && python -m unittest discover -s tests -v"""
import ast
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
TMP = tempfile.mkdtemp(prefix="iwdd_test_")
os.environ["IWDD_RUNS"] = TMP            # i test non scrivono nei risultati veri

from iwdd import evaluate as ev                                    # noqa: E402
from iwdd.config import load_config                                # noqa: E402
from iwdd.data import Video, shard, stratified_limit              # noqa: E402
from iwdd.extract import LONG_COLS                                 # noqa: E402
from iwdd.features import video_matrix, window_matrix, window_probabilities   # noqa: E402
from iwdd.paths import PROJECT_ROOT, Run                           # noqa: E402
from iwdd.questions import context_text, load_derived, load_questions, questions_hash   # noqa: E402


def project_evaluate():
    """evaluate() di main.py del progetto, estratta con ast (main.py importa torch e non va toccato)."""
    tree = ast.parse((PROJECT_ROOT / "main.py").read_text(encoding="utf-8"))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            exec(compile(ast.Module([node], []), "main.py", "exec"), ns)
    return ns["evaluate"]


class TestQuestions(unittest.TestCase):
    def test_set_bloccato(self):
        qs = load_questions()
        self.assertEqual(len(qs), 35)
        self.assertEqual(sum(q.kind == "yesno" for q in qs), 26)
        self.assertEqual(sum(q.kind == "claim" for q in qs), 8)
        self.assertEqual(sum(q.kind == "multi" for q in qs), 1)
        self.assertEqual(sum(q.origin == "utente" for q in qs), 24)
        self.assertEqual(sum(q.origin == "suggerita" for q in qs), 11)
        for q in qs:
            self.assertTrue(q.en.strip() and q.it.strip(), q.id)
            if q.kind == "claim":
                self.assertEqual(q.n_options, 3)
                self.assertEqual(q.jev_item()[2], "claim")
            if q.kind == "multi":
                self.assertEqual(q.n_options, 4)
                self.assertEqual(len(q.options_it), 4)
        self.assertEqual(len({q.id for q in qs}), 35)

    def test_derived(self):
        ids = {q.id for q in load_questions()}
        for spec in load_derived().values():
            self.assertTrue(set(spec["plus"] + spec["minus"]) <= ids)

    def test_hash_e_contesto(self):
        qs = load_questions()
        self.assertEqual(questions_hash(qs, "rows", 1003520), questions_hash(qs, "rows", 1003520))
        self.assertNotEqual(questions_hash(qs, "rows", 1003520), questions_hash(qs, "none", 1003520))
        self.assertEqual(context_text("none", 0.5), "")
        self.assertIn("top row shows the first three frames", context_text("rows", 0.5))
        self.assertNotIn("top row shows", context_text("basic", 0.5))
        self.assertIn("0.5 seconds", context_text("basic", 0.5))


class TestEvaluate(unittest.TestCase):
    def test_come_main_evaluate(self):
        f = project_evaluate()
        rng = random.Random(0)
        for _ in range(2000):
            pos = rng.random() < 0.5
            g = rng.randint(0, 40)
            ans = None if rng.random() < 0.3 else rng.randint(0, 70)
            lab = {"label": "true" if pos else "false", "timestamp": g if pos else -1}
            self.assertEqual(ev.outcome(pos, g if pos else -1, ans), f(lab, None if ans is None else float(ans)))

    def test_risposta_intera_e_cap(self):
        self.assertEqual(float(ev.answer_int(5.75, 2, 100)), 8.0)       # 7.75 -> 8
        self.assertEqual(float(ev.answer_int(70.5, 2, 60)), 60.0)       # non oltre la fine del video
        self.assertEqual(float(ev.answer_int(0.0, -2, 10)), 0.0)        # mai negativa

    def test_prf(self):
        r = ev.from_outcomes(["TP"] * 6 + ["FP"] * 3 + ["FN"] * 2 + ["TN"] * 5)
        self.assertAlmostEqual(r["precision"], 6 / 9)
        self.assertAlmostEqual(r["recall"], 6 / 8)
        self.assertAlmostEqual(r["f1"], 2 * 6 / (2 * 6 + 3 + 2))

    def test_grid_search(self):
        is_pos = np.array([True, True, False, False])
        g = np.array([10, 20, -1, -1], float)
        dur = np.array([40, 40, 40, 40], float)
        p = np.array([0.9, 0.8, 0.3, 0.2])
        best_t = {0: np.array([10.0, 21.0, 12.0, 5.0])}
        f1, d, thr, _ = ev.grid_search(p, best_t, dur, is_pos, g, [0])
        self.assertEqual((f1, d), (1.0, 0))
        self.assertAlmostEqual(thr, 0.8)


class TestData(unittest.TestCase):
    def test_shard_bilanciato_e_completo(self):
        vs = [Video(f"vid{i:04d}", "train", i % 2 == 0, 5.0, "normal", 10.0, "day", "1x1") for i in range(23)]
        w = {v.vid: (i % 7) + 1 for i, v in enumerate(vs)}
        parts = [shard(vs, k, 4, w) for k in range(4)]
        self.assertEqual(sorted(v.vid for p in parts for v in p), sorted(v.vid for v in vs))
        loads = [sum(w[v.vid] for v in p) for p in parts]
        self.assertLessEqual(max(loads) - min(loads), 7)

    def test_limit_stratificato(self):
        vs = ([Video(f"n{i}", "train", False, -1, "normal", 1, "day", "") for i in range(50)] +
              [Video(f"s{i}", "train", True, 3, "static dumping", 1, "day", "") for i in range(25)] +
              [Video(f"d{i}", "train", True, 3, "dynamic dumping", 1, "day", "") for i in range(25)])
        sub = stratified_limit(vs, 20, 0)
        self.assertEqual([sum(v.event_type == t for v in sub) for t in ("normal", "static dumping", "dynamic dumping")], [10, 5, 5])
        self.assertEqual(sub, stratified_limit(vs, 20, 0))


def synth_long(qs, vid, n_win, signal, rng):
    rows = []
    for w in range(n_win):
        for q in qs:
            l = np.full(4, np.nan, dtype=np.float32)
            base = rng.normal(0, 1.0, q.n_options)
            if signal and w == 0 and q.id in ("c_put_leave", "waste_put_down", "walking_away", "scene_summary"):
                base[0] += 5.0
            if q.kind == "yesno":
                base = np.array([base[0] + 20, base[1] + 20])
            l[:q.n_options] = base + 25
            rows.append((vid, w, q.id, q.n_options, *l))
    return pd.DataFrame(rows, columns=LONG_COLS).astype({"win": "int64", "n_opt": "int64"})


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.qs, self.derived = load_questions(), load_derived()
        rng = np.random.default_rng(0)
        self.long = pd.concat([synth_long(self.qs, "vidA", 3, True, rng), synth_long(self.qs, "vidB", 2, False, rng)], ignore_index=True)

    def test_window_matrix(self):
        X = window_matrix(self.long, self.qs, self.derived)
        self.assertEqual(len(X), 5)
        self.assertEqual(X.shape[1], 26 + 8 + 3 + 2)                      # sì/no + affermazioni + scelta (3) + 2 derivate
        self.assertTrue((X.abs().max() <= 16.0).all())
        self.assertIn("dep_state", X.columns)
        d = X["dep_state"]
        self.assertTrue(np.allclose(d.values, (X.hold_first - X.hold_last + X.ground_last - X.ground_first).clip(-16, 16).values))

    def test_video_matrix(self):
        X = window_matrix(self.long, self.qs, self.derived)
        F = video_matrix(X, ["vidA", "vidB", "vidC"], ("max", "mean", "top3", "frac"))
        self.assertEqual(F.shape[1], X.shape[1] * 4 + 1)
        self.assertEqual(list(F.no_events), [0.0, 0.0, 1.0])
        self.assertTrue(F.loc["vidC"].drop("no_events").isna().all())
        self.assertLessEqual(F.loc["vidA", "carry_waste|mean"], F.loc["vidA", "carry_waste|max"])

    def test_probabilita(self):
        P = window_probabilities(self.long, self.qs)
        self.assertTrue(((P >= 0) & (P <= 1)).all().all())
        claim = [c for c in P.columns if c.startswith("c_put_leave__")]
        self.assertEqual(len(claim), 3)
        self.assertTrue(np.allclose(P[claim].sum(axis=1), 1.0, atol=2e-3))


class TestEndToEndSintetico(unittest.TestCase):
    """Addestramento + valutazione + report su feature finte (nessuna GPU): verifica che tutto giri e produca i file."""

    def test_run(self):
        from iwdd.report import write_report
        from iwdd.train_eval import run_train_eval
        cfg = load_config()
        cfg["cv"]["repeats"] = 1
        cfg["bootstrap"] = 50
        cfg["video_model"]["C_grid"] = [0.03, 0.3]
        cfg["video_model"]["aggs_options"] = [["max"], ["max", "mean"]]
        qs, derived = load_questions(), load_derived()
        run = Run("sintetico")
        rng = np.random.default_rng(1)
        pyr = random.Random(1)
        vids, meta, groups = [], {}, {}
        for split, n in (("train", 60), ("test", 24)):
            for i in range(n):
                pos = i % 2 == 0
                vid = f"{'t' if split == 'train' else 'x'}{i:03d}"
                g = float(pyr.randint(2, 20)) if pos else -1.0
                dur = 30.0
                vids.append(Video(vid, split, pos, g, ("static dumping" if i % 4 == 0 else "dynamic dumping") if pos else "normal", dur, "day", "1280x720"))
                n_win = pyr.randint(2, 8)
                long = synth_long(qs, vid, n_win, pos, rng)
                # la griglia 0 e' quella dell'evento: i tempi sono consecutivi, la 0 contiene g
                ts = [max(0.0, g - 2) if pos else 0.0] + [4.0 * (k + 6) for k in range(n_win - 1)]
                wins = pd.DataFrame({"video": vid, "win": range(n_win), "t_start": ts, "t_center": [t + 1.9 for t in ts],
                                     "t_end": [t + 3.75 for t in ts], "image": [f"{vid}_e{k}.jpg" for k in range(n_win)],
                                     "image_tokens": 900, "t_win": 1.0})
                long.to_parquet(run.features / f"{vid}.parquet", index=False)
                wins.to_parquet(run.features / f"{vid}.windows.parquet", index=False)
                meta[vid] = dict(width=1280, height=720, t_last=dur, fps_mean=25.0, brightness=100.0,
                                 bg_signature=list(rng.normal(size=576)))
                (run.meta / f"{vid}.json").write_text(json.dumps(meta[vid]))
                groups[vid] = len(groups)
        (run.meta / "groups.json").write_text(json.dumps({"sim": 0.95, "group": groups}))
        (run.features / "_manifest.json").write_text(json.dumps(
            {"questions_hash": questions_hash(qs, "rows", 1003520), "context_mode": "rows", "max_pixels": 1003520,
             "question_ids": [q.id for q in qs]}))
        tr = [v for v in vids if v.split == "train"]
        te = [v for v in vids if v.split == "test"]
        s = run_train_eval(run, cfg, qs, derived, tr, te)
        self.assertGreater(s["test"]["f1_challenge"], 0.5)
        self.assertGreater(s["test"]["auc_detect"], 0.8)
        self.assertEqual(s["test"]["TP"] + s["test"]["FN"] + s["test"]["FP"] - (s["test"]["n"] - s["test"]["n_pos"] - s["test"]["TN"]) >= 0, True)
        for rel in ("metrics/summary.json", "metrics/video_model_grid.csv", "tables/predictions_test.csv", "tables/question_stats.csv",
                    "tables/probabilities_grids.csv", "models/default/final_config.json", "models/default/video_model.joblib"):
            self.assertTrue((run.root / rel).exists(), rel)
        self.assertTrue(write_report(run, qs).exists())
        # il modello salvato si ricarica da final_config.json e da' le stesse probabilita'
        from iwdd.models import LR
        cfgm = json.loads((run.root / "models/default/final_config.json").read_text())
        import joblib
        m1 = joblib.load(run.root / "models/default/video_model.joblib")
        m2 = LR.from_dict(cfgm["video_model"])
        X = pd.DataFrame(rng.normal(size=(5, len(m1.cols))), columns=m1.cols)
        self.assertTrue(np.allclose(m1.predict_proba(X), m2.predict_proba(X)))


class TestRobustezzaFile(unittest.TestCase):
    def test_config_e_domande_con_bom(self):
        """Su Windows un editor o PowerShell possono aggiungere il BOM: la lettura deve funzionare lo stesso."""
        import codecs
        import shutil
        d = Path(tempfile.mkdtemp())
        for name in ("default.json", "questions.json", "questions_it.json"):
            raw = (Path(__file__).resolve().parents[1] / "config" / name).read_bytes()
            (d / name).write_bytes(codecs.BOM_UTF8 + raw)
        self.assertEqual(load_config(d / "default.json")["jev"]["chunk"], 8)
        qs = load_questions(d / "questions.json")
        self.assertEqual(len(qs), 35)
        self.assertTrue(all(q.it for q in qs))            # anche l'italiano (file gemello) si legge con il BOM
        shutil.rmtree(d, ignore_errors=True)

    def test_default_sicuri_per_8gb(self):
        cfg = load_config()
        self.assertEqual(cfg["jev"]["chunk"], 8)           # con 35 domande e ~980 token 11 va in spill su 8 GB (misurato)
        self.assertEqual(cfg["jev"]["max_pixels"], 1003520)
        self.assertEqual(cfg["jev"]["context_mode"], "rows")


if __name__ == "__main__":
    unittest.main()
