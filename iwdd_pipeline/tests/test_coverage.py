"""Griglie di copertura (preprocessing_pt2.py) dentro 'prepare' e indicatore is_fill nel modello a finestre (nessuna GPU)."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("IWDD_RUNS", tempfile.mkdtemp(prefix="iwdd_cov_runs_"))
os.environ.setdefault("IWDD_CLASSIFIER_OUT", tempfile.mkdtemp(prefix="iwdd_cov_out_"))

from iwdd import events                                                  # noqa: E402
from iwdd.events import grid_kind, list_event_images                     # noqa: E402
from iwdd.paths import Run, ensure_project_imports                       # noqa: E402

ensure_project_imports()
import decompose_video as dv_mod                                         # noqa: E402
import preprocessing as pp_mod                                           # noqa: E402
import preprocessing_pt2 as pt2_mod                                      # noqa: E402

N_FRAMES = 40          # 10 s a 4 fps


class TestPrepareConCopertura(unittest.TestCase):
    """decompose_video e preprocess_video finti (veloci), fill_coverage VERO sui frame finti."""

    def setUp(self):
        self.orig = (dv_mod.decompose_video, pp_mod.preprocess_video, pt2_mod.fill_coverage)
        self.events_by_video = {"vidA": [(0, 15)], "vidB": []}       # vidB: nessun soggetto tracciato -> nessuna griglia evento
        self.seen_final_during_fill = []

        def fake_decompose(path, fps_out=4, output_root=None):
            out = Path(output_root) / Path(path).stem
            out.mkdir(parents=True, exist_ok=True)
            for i in range(N_FRAMES):
                Image.new("RGB", (64, 48), (i * 6 % 255, 80, 120)).save(out / f"frame_{i:04d}.jpg")
            return sorted(out.glob("*.jpg"))

        def fake_preprocess(frames_dir, strips_dir, debug_root=None, fps=4.0, jpeg_quality=95):
            video, strips_dir = Path(frames_dir).name, Path(strips_dir)
            tmp = strips_dir.with_name(strips_dir.name + ".tmp")          # come il vero: .tmp e rename
            tmp.mkdir(parents=True)
            names = []
            for k, (a, b) in enumerate(self.events_by_video[video]):
                names.append(f"{video}_e{k:03d}_t{a}_t{b}.jpg")
                Image.new("RGB", (96, 72), (200, 0, 0)).save(tmp / names[-1])
            tmp.rename(strips_dir)
            return [strips_dir / n for n in names]

        def watched_fill(frames_dir, strips_dir, **kw):
            self.seen_final_during_fill.append((self.run.events / Path(frames_dir).name).exists())
            return self.orig[2](frames_dir, strips_dir, **kw)

        dv_mod.decompose_video, pp_mod.preprocess_video, pt2_mod.fill_coverage = fake_decompose, fake_preprocess, watched_fill

    def tearDown(self):
        dv_mod.decompose_video, pp_mod.preprocess_video, pt2_mod.fill_coverage = self.orig

    def _prepare(self, run_name, vid, fill):
        self.run = Run(run_name)
        shutil.rmtree(self.run.root, ignore_errors=True)
        return events._prepare_job((run_name, "train", vid, (), 4, 95, fill))

    def test_buchi_riempiti_prima_della_cartella_finale(self):
        vid, msg, _ = self._prepare("cov_a", "vidA", 2.0)
        self.assertFalse(msg.startswith("ERRORE"), msg)
        self.assertEqual(self.seen_final_during_fill, [False])            # extract --wait-for non vede mai la cartella a meta'
        d = self.run.events / "vidA"
        self.assertFalse((self.run.events / "vidA.prep").exists())
        kinds = [grid_kind(p.name) for _, _, p in list_event_images(d)]
        self.assertEqual(kinds.count("event"), 1)
        self.assertEqual(kinds.count("fill"), 2)                          # buco 16..39 (6 s): finestre da 4 s, passo 2 s
        cov = json.loads((d / "coverage.json").read_text())
        self.assertEqual(cov["reason"], "gaps")
        self.assertEqual(len(cov["gaps_filled"]), 1)
        covered = set()
        for a, b, _ in list_event_images(d):
            covered.update(range(a, b + 1))
        self.assertEqual(covered, set(range(N_FRAMES)))                  # nessun frame scoperto

    def test_video_senza_eventi_coperto_tutto(self):
        _, msg, _ = self._prepare("cov_b", "vidB", 2.0)
        self.assertIn("no_grids", msg)
        imgs = list_event_images(self.run.events / "vidB")
        self.assertGreater(len(imgs), 0)
        self.assertTrue(all(grid_kind(p.name) == "fill" for _, _, p in imgs))

    def test_copertura_disattivata(self):
        _, msg, _ = self._prepare("cov_c", "vidA", None)
        d = self.run.events / "vidA"
        self.assertEqual([grid_kind(p.name) for _, _, p in list_event_images(d)], ["event"])
        self.assertFalse((d / "coverage.json").exists())
        self.assertEqual(self.seen_final_during_fill, [])


class TestIsFill(unittest.TestCase):
    def test_solo_nel_modello_a_finestre(self):
        from iwdd.config import load_config
        from iwdd.questions import load_derived, load_questions
        from iwdd.train_eval import run_train_eval
        from test_classifiers import synthetic_run
        vids = synthetic_run("sint_fill")
        run = Run("sint_fill")
        for v in vids:                                                     # l'ultima griglia di ogni video diventa di copertura
            p = run.features / f"{v.vid}.windows.parquet"
            w = pd.read_parquet(p)
            w["kind"] = ["fill" if k == w.win.max() else "event" for k in w.win]
            w.to_parquet(p, index=False)
        cfg = load_config()
        cfg["cv"]["repeats"] = 1
        cfg["bootstrap"] = 20
        cfg["video_model"]["C_grid"] = [0.1]
        cfg["video_model"]["aggs_options"] = [["max"]]
        cfg["window_model"]["C_grid"] = [0.1]
        cfg["window_model"]["deltas"] = [0, 2]
        s = run_train_eval(run, cfg, load_questions(), load_derived(), [v for v in vids if v.split == "train"],
                           [v for v in vids if v.split == "test"])
        final = json.loads((run.models / "default" / "final_config.json").read_text())
        self.assertIn("is_fill", final["window_model"]["features"])
        self.assertFalse(any(f.startswith("is_fill") for f in final["video_model"]["features"]))
        self.assertGreater(s["train"]["n_fill_grids"], 0)


if __name__ == "__main__":
    unittest.main()
