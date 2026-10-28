"""Collaudo del codice di estrazione con uno scorer FINTO (nessuna GPU): scrittura dei parquet, tempi delle griglie dai nomi file,
ripresa, shard, budget di tempo, manifest, video senza immagini."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["IWDD_RUNS"] = tempfile.mkdtemp(prefix="iwdd_fake_")

import iwdd.jev as jev                                      # noqa: E402
from iwdd.config import load_config                          # noqa: E402
from iwdd.data import Video                                  # noqa: E402
from iwdd.extract import extract                             # noqa: E402
from iwdd.features import read_features                     # noqa: E402
from iwdd.paths import Run                                   # noqa: E402
from iwdd.questions import load_questions                    # noqa: E402


class FakeScorer:
    def __init__(self):
        self.calls, self.contexts = 0, []
        self.jev_main = type("M", (), {"crop_margins": staticmethod(lambda im: im)})()

    def score_raw(self, image, questions, context="", chunk=11):
        self.calls += 1
        self.contexts.append(context)
        out = np.full((len(questions), 4), np.nan, dtype=np.float32)
        for i, q in enumerate(questions):
            n = len(q[1]) if q[2:] == () else 3
            out[i, :n] = np.arange(n) + i
        return out, {"image_tokens": 900, "t_prepare": 0.0, "t_prefill": 0.0, "t_suffix": 0.0}


def make_video(run, vid, windows):
    d = run.events / vid
    d.mkdir(parents=True, exist_ok=True)
    for k, (a, b) in enumerate(windows):
        Image.new("RGB", (64, 48), (k * 10 % 255, 0, 0)).save(d / f"{vid}_e{k:03d}_t{a}_t{b}.jpg")
    return Video(vid, "train", True, 5.0, "static dumping", 20.0, "day", "64x48")


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.fake = FakeScorer()
        jev.build_scorer = lambda max_pixels: self.fake       # extract() importa build_scorer a ogni chiamata
        self.qs = load_questions()
        self.cfg = load_config()
        self.run = Run(f"fake_{self._testMethodName}")
        shutil.rmtree(self.run.root, ignore_errors=True)
        self.vids = [make_video(self.run, f"vid{i:04d}", [(0, 15), (4, 19), (8, 23)][: 1 + i % 3]) for i in range(6)]

    def test_scrive_e_riprende(self):
        r = extract(self.run, self.vids, self.qs, self.cfg)
        self.assertEqual((r["done"], r["remaining"], r["errors"], r["missing"]), (6, 0, 0, 0))
        long, wins, have = read_features(self.run, [v.vid for v in self.vids])
        self.assertEqual(len(have), 6)
        n_img = sum(1 + i % 3 for i in range(6))
        self.assertEqual(len(wins), n_img)
        self.assertEqual(len(long), n_img * 35)
        self.assertEqual(self.fake.calls, n_img)
        w = wins[wins.video == "vid0002"].sort_values("win")
        self.assertEqual(list(w.t_end), [15 / 4, 19 / 4, 23 / 4])                    # tempo = indice / 4
        self.assertTrue(np.allclose(w.t_center, (w.t_start + w.t_end) / 2))
        self.assertEqual(set(long.qid), {q.id for q in self.qs})
        # ripresa: nessun lavoro in piu'
        r2 = extract(self.run, self.vids, self.qs, self.cfg)
        self.assertEqual(r2["done"], 0)
        self.assertEqual(self.fake.calls, n_img)

    def test_contesto(self):
        extract(self.run, self.vids[:1], self.qs, self.cfg)
        self.assertIn("The top row shows the first three frames", self.fake.contexts[0])
        r = Run("fake_none")
        shutil.rmtree(r.root, ignore_errors=True)
        v = make_video(r, "vid0100", [(0, 15)])
        f2 = FakeScorer()
        jev.build_scorer = lambda max_pixels: f2
        extract(r, [v], self.qs, self.cfg, context_mode="none")
        self.assertEqual(f2.contexts, [""])

    def test_in_parallelo_a_prepare(self):
        """--wait-for: i video compaiono uno alla volta mentre 'prepare' lavora (cartella scritta in .tmp e rinominata); extract li
        estrae appena pronti e si ferma quando c'e' il file segnale e non resta nulla. Un video mai preparato risulta 'missing'."""
        import threading
        import time
        hold = self.run.root / "hold"
        hold.mkdir()
        late = self.vids[3:]
        for v in late:                                   # tolti: li "prepara" il thread qui sotto, con calma
            shutil.move(str(self.run.events / v.vid), str(hold / v.vid))
        ghost = Video("vid9999", "train", False, -1.0, "normal", 10.0, "day", "64x48")    # prepare fallito: niente cartella
        done_flag = self.run.root / "prepare.done"

        def fake_prepare():
            for v in late:
                time.sleep(0.3)
                tmp = self.run.events / (v.vid + ".tmp")
                shutil.move(str(hold / v.vid), str(tmp))
                tmp.rename(self.run.events / v.vid)
            time.sleep(0.3)
            done_flag.touch()

        th = threading.Thread(target=fake_prepare)
        th.start()
        r = extract(self.run, self.vids + [ghost], self.qs, self.cfg, wait_for=str(done_flag), poll_s=0.05)
        th.join()
        self.assertEqual((r["done"], r["errors"], r["missing"]), (6, 0, 1))
        self.assertEqual(self.fake.calls, sum(1 + i % 3 for i in range(6)))
        _, _, have = read_features(self.run, [v.vid for v in self.vids])
        self.assertEqual(len(have), 6)

    def test_tipo_di_griglia(self):
        """Le griglie di copertura (_f) si estraggono come le altre e sono marcate kind='fill' nei tempi delle griglie."""
        v = self.vids[0]
        Image.new("RGB", (64, 48), (0, 90, 0)).save(self.run.events / v.vid / f"{v.vid}_f000_t20_t35.jpg")
        extract(self.run, [v], self.qs, self.cfg)
        _, wins, _ = read_features(self.run, [v.vid])
        self.assertEqual(sorted(wins.kind), ["event", "fill"])
        self.assertEqual(wins.set_index("image").loc[f"{v.vid}_f000_t20_t35.jpg", "kind"], "fill")

    def test_shard_coprono_tutto_senza_sovrapposizioni(self):
        r0 = extract(self.run, self.vids, self.qs, self.cfg, shard_idx=0, num_shards=2)
        r1 = extract(self.run, self.vids, self.qs, self.cfg, shard_idx=1, num_shards=2)
        self.assertEqual(r0["done"] + r1["done"], 6)
        self.assertEqual(self.fake.calls, sum(1 + i % 3 for i in range(6)))

    def test_budget_di_tempo(self):
        import time
        orig = self.fake.score_raw

        def slow(*a, **k):
            time.sleep(0.05)
            return orig(*a, **k)
        self.fake.score_raw = slow
        r = extract(self.run, self.vids, self.qs, self.cfg, max_minutes=0.0005)     # 0,03 s: si ferma dopo il primo video
        self.assertGreater(r["remaining"], 0)
        self.assertGreaterEqual(r["done"], 1)
        r2 = extract(self.run, self.vids, self.qs, self.cfg)
        self.assertEqual(r["done"] + r2["done"], 6)

    def test_manifest_impedisce_di_mescolare(self):
        extract(self.run, self.vids[:2], self.qs, self.cfg)
        with self.assertRaises(RuntimeError):
            extract(self.run, self.vids, self.qs, self.cfg, context_mode="none")
        extract(self.run, self.vids, self.qs, self.cfg, context_mode="none", force=True)      # con --force si puo'

    def test_quant_bits_non_si_mescola(self):
        extract(self.run, self.vids[:2], self.qs, self.cfg)
        os.environ["QUANT_BITS"] = "4"
        try:
            with self.assertRaises(RuntimeError):
                extract(self.run, self.vids, self.qs, self.cfg)
        finally:
            os.environ.pop("QUANT_BITS", None)

    def test_video_senza_immagini_e_video_vuoto(self):
        vuoto = Video("vid0900", "train", False, -1.0, "normal", 10.0, "day", "")
        (self.run.events / "vid0900").mkdir(parents=True)                                  # cartella presente ma senza griglie
        assente = Video("vid0901", "train", False, -1.0, "normal", 10.0, "day", "")
        r = extract(self.run, [vuoto, assente], self.qs, self.cfg)
        self.assertEqual((r["done"], r["missing"]), (1, 1))
        long, wins, have = read_features(self.run, ["vid0900", "vid0901"])
        self.assertEqual(have, {"vid0900"})
        self.assertEqual((len(long), len(wins)), (0, 0))

    def test_audit_domande_nel_run(self):
        extract(self.run, self.vids[:1], self.qs, self.cfg)
        self.assertTrue((self.run.root / "questions_used.json").exists())
        txt = (self.run.root / "prompts_used.txt").read_text(encoding="utf-8")
        self.assertTrue(txt.isascii())
        self.assertIn("Question: Is a person carrying a piece of waste?", txt)


if __name__ == "__main__":
    unittest.main()
