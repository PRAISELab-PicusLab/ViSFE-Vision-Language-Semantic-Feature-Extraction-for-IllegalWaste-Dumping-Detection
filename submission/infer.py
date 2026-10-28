"""IWDD 2027 - inference engine of the WasteVision submission.

For every test video it applies, step by step and with the very same code used for training/validation, the pipeline described in
README.md:

  1. CPU   decompose_video (4 fps) -> preprocessing.preprocess_video (tracking, 3x3 grids of 9 frames, zoom on the subject)
           -> preprocessing_pt2.fill_coverage (full-frame grids on the stretches no event grid covers)
  2. GPU   Visual-JEV (Qwen3-VL-4B 8-bit + LoRA adapter) answers 35 fixed questions on every grid -> raw logits (iwdd.jev)
  3. CPU   logits -> per-grid signals (iwdd.features.window_matrix) -> per-video aggregates (max/mean/top3/frac)
  4.       two logistic regressions: the video model says "is there a dumping?", the window model scores each grid; the answer is
           the end time of the best grid + 2 s, an integer number of seconds not beyond the end of the video.

The CPU preparation of the next videos runs in worker processes while the GPU scores the current one.
"""
import contextlib
import csv
import math
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("IWDD_PROJECT", str(ROOT))           # iwdd.paths: where main.py, preprocessing.py and visual-jev/ live
os.environ.setdefault("QUANT_BITS", "8")                   # the validated setting (8-bit backbone)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json                                                # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

BASE_REPO = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"  # pinned: the exact weights the model was validated with
DEFAULT_CHUNK = 40      # questions scored per GPU block. 40 (= all 35 questions at once) is the setting the model was validated with: it
                        # reproduces the validation outputs exactly (peak ~11.3 GiB on an A100). Smaller blocks use less VRAM (8 -> ~6.3 GiB)
                        # but change the GPU logits slightly (bf16), which can flip borderline decisions.
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv", ".flv")
LOG_COLUMNS = ["video", "status", "n_grids", "n_fill_grids", "p_video", "threshold", "answer", "t_last_s", "prepare_s", "gpu_s"]


# --------------------------------------------------------------------------------------------------------------------- helpers
def list_videos(folder):
    """Video files of `folder` (sorted). Falls back to every regular file if no known video extension is present."""
    files = sorted(p for p in Path(folder).iterdir() if p.is_file() and not p.name.startswith("."))
    known = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    return known or files


def video_t_last(path):
    """Timestamp (s) of the last frame, read exactly like the scan step used in training (it is the clip duration the answer is capped to)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    last = 0.0
    while cap.grab():
        last = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cap.release()
    return float(last)


def answer_seconds(t, delta, dur):
    """Same rule as iwdd.evaluate.answer_int: t + delta, not beyond the end of the video, rounded to the nearest second, never negative."""
    return int(max(0, math.floor(min(float(t) + delta, float(dur)) + 0.5)))


class LinearModel:
    """A fitted logistic regression (imputation with the training mean, standardisation, linear score, sigmoid) using only numpy."""

    def __init__(self, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.features = d["features"]
        self.mu, self.mean, self.std = (np.asarray(d[k], dtype=float) for k in ("impute_mean", "scale_mean", "scale_std"))
        self.coef, self.intercept = np.asarray(d["coef"], dtype=float), float(d["intercept"])

    def predict(self, frame):
        A = frame.reindex(columns=self.features).to_numpy(dtype=float)
        A = np.where(np.isnan(A), self.mu, A)
        return 1.0 / (1.0 + np.exp(-(((A - self.mean) / self.std) @ self.coef + self.intercept)))


def ensure_base_model():
    """Local path of the Qwen3-VL-4B-Instruct snapshot (downloaded once from the Hugging Face Hub, pinned revision)."""
    override = os.environ.get("IWDD_BASE_MODEL")
    if override:
        return override
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(BASE_REPO, revision=BASE_REVISION)
    except Exception as e:
        raise RuntimeError(f"cannot get {BASE_REPO}@{BASE_REVISION[:7]} from the Hugging Face Hub ({type(e).__name__}: {e}). "
                           "It needs internet access (or a populated HF_HOME with HF_HUB_OFFLINE=1).") from e


# --------------------------------------------------------------------------------------------------------------------- CPU stage
def prepare_video(job):
    """Worker process: video -> event grids + coverage grids in a private temporary folder. Never touches the GPU."""
    idx, video_path, work_root, fps, quality, fill_gaps_s = job
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("IWDD_PROJECT", str(ROOT))
    t0 = time.perf_counter()
    tmp = Path(work_root) / f"v{idx:05d}"
    try:
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):      # preprocessing.py prints per-video diagnostics
            from decompose_video import decompose_video
            from preprocessing import preprocess_video
            from preprocessing_pt2 import fill_coverage
            t_last = video_t_last(video_path)
            frames_root, events = tmp / "frames", tmp / "events"
            decompose_video(Path(video_path), fps_out=fps, output_root=frames_root)
            frames = frames_root / Path(video_path).stem
            preprocess_video(frames, events, debug_root=None, fps=fps, jpeg_quality=quality)
            cov = fill_coverage(frames, events, debug_root=None, fps=fps, gap_s=fill_gaps_s, jpeg_quality=quality)
            shutil.rmtree(frames_root, ignore_errors=True)
        return dict(idx=idx, ok=True, events=str(events), t_last=t_last, n_fill=len(cov["fill_grids"]), secs=time.perf_counter() - t0)
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return dict(idx=idx, ok=False, error=f"{type(e).__name__}: {e}", secs=time.perf_counter() - t0)


# --------------------------------------------------------------------------------------------------------------------- the detector
class Detector:
    def __init__(self, chunk=None):
        from iwdd.config import load_config
        from iwdd.questions import load_derived, load_questions, questions_hash
        self.cfg = load_config()
        self.model = json.loads((ROOT / "models" / "final_config.json").read_text(encoding="utf-8"))
        self.questions, self.derived = load_questions(), load_derived()
        m, jev = self.model, self.cfg["jev"]
        # the model must be applied to exactly the same questions / context / resolution it was trained on
        if [q.id for q in self.questions] != m["question_ids"] or questions_hash(self.questions, jev["context_mode"], jev["max_pixels"]) != m["questions_hash"]:
            raise RuntimeError("the questions / context / resolution in config/ differ from the ones the model was trained with")
        if jev["context_mode"] != m["context_mode"] or jev["max_pixels"] != m["max_pixels"]:
            raise RuntimeError("config/default.json and models/final_config.json disagree on context mode or resolution")
        self.fps, self.quality, self.fill_gaps_s = (self.cfg["events"][k] for k in ("fps", "jpeg_quality", "fill_gaps_s"))
        self.chunk = int(chunk or DEFAULT_CHUNK)
        self.video_model = LinearModel(ROOT / "models" / "video_model.json")
        self.window_model = LinearModel(ROOT / "models" / "window_model.json")
        self.threshold, self.delta, self.time_col = float(m["threshold"]), int(m["delta"]), m["time_col"]
        self.aggs = tuple(m["aggs"])
        self.scorer = None

    def load_gpu(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU available: in Colab use Runtime > Change runtime type > GPU (T4 or better)")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1024 ** 3:.0f} GiB, compute capability {props.major}.{props.minor}) | chunk {self.chunk}", flush=True)
        base = ensure_base_model()
        from iwdd.paths import ensure_project_imports
        ensure_project_imports()
        import main as project_main                              # the project's main.py: ChoiceScorer = Qwen3-VL 8-bit + LoRA adapter
        project_main.BASE_MODEL = base                           # local snapshot at the pinned revision
        project_main.ADAPTER = str(ROOT / "models" / "visual-jev-adapter")
        from iwdd.jev import build_scorer
        self.scorer = build_scorer(self.cfg["jev"]["max_pixels"])
        self.qlist = [q.jev_item() for q in self.questions]

    # -- GPU stage + decision
    def decide(self, vid, events_dir, t_last):
        """Visual-JEV on every grid, then the two logistic models. Returns a dict with the decision."""
        from PIL import Image
        from iwdd.events import grid_kind, list_event_images
        from iwdd.features import LONG_COLS, base_columns, video_matrix, window_matrix
        from iwdd.questions import context_text
        crop_margins = self.scorer.jev_main.crop_margins
        grids = list_event_images(events_dir)
        if not grids:
            return dict(n_grids=0, n_fill=0, p_video=float("nan"), answer=None)
        rows, kinds, t_end = [], [], []
        for k, (a, b, path) in enumerate(grids):
            with Image.open(path) as src:
                image = crop_margins(src.convert("RGB"))
            dt = (b - a) / self.fps / 8.0 if b > a else 0.5
            lg, _ = self.scorer.score_raw(image, self.qlist, context=context_text(self.cfg["jev"]["context_mode"], dt), chunk=self.chunk)
            t_end.append(b / self.fps)
            kinds.append(grid_kind(path.name))
            for qi, q in enumerate(self.questions):
                logits = np.full(4, np.nan, dtype=np.float32)
                logits[:q.n_options] = lg[qi, :q.n_options]
                rows.append((vid, k, q.id, q.n_options, *logits))
        long = pd.DataFrame(rows, columns=LONG_COLS).astype({"win": "int64", "n_opt": "int64", "l0": "float32", "l1": "float32", "l2": "float32", "l3": "float32"})
        X = window_matrix(long, self.questions, self.derived, self.cfg["clip_logit"], self.cfg["clip_derived"])
        cols = base_columns(X, self.derived)
        p_video = float(self.video_model.predict(video_matrix(X, [vid], self.aggs, cols))[0])
        Xw = X[cols].assign(is_fill=[1.0 if k == "fill" else 0.0 for k in kinds])
        scores = self.window_model.predict(Xw)
        best = int(np.argmax(scores))                             # first maximum, like the validation code
        answer = answer_seconds(t_end[best], self.delta, t_last) if p_video >= self.threshold else None
        return dict(n_grids=len(grids), n_fill=sum(k == "fill" for k in kinds), p_video=p_video, answer=answer, best_grid=best, best_score=float(scores[best]))


# --------------------------------------------------------------------------------------------------------------------- main loop
def write_result(results_dir, video_name, answer):
    """`<video file name>.txt` containing the detection second, or an empty file when no dumping is detected (as in the example)."""
    with open(os.path.join(results_dir, video_name + ".txt"), "w") as f:
        if answer is not None:
            f.write(str(int(answer)))


def run(videos_dir, results_dir, chunk=None, resume=False, workers=None, work_dir=None, log_path="submission_log.csv", download_only=False):
    if download_only:
        print("base model:", ensure_base_model())
        return 0
    os.makedirs(results_dir, exist_ok=True)
    videos = list_videos(videos_dir)
    if resume:
        videos = [v for v in videos if not os.path.exists(os.path.join(results_dir, v.name + ".txt"))]
    print(f"{len(videos)} video(s) to process from {videos_dir} -> {results_dir}", flush=True)
    if not videos:
        return 0
    det = Detector(chunk=chunk)
    workers = int(workers or max(1, min(4, (os.cpu_count() or 2) - 1)))
    work_root = tempfile.mkdtemp(prefix="iwdd_work_", dir=work_dir)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))      # created before CUDA is initialised
    jobs = {}

    def submit(i):
        if i < len(videos) and i not in jobs:
            jobs[i] = pool.submit(prepare_video, (i, str(videos[i]), work_root, det.fps, det.quality, det.fill_gaps_s))

    t_start, n_pos, log = time.perf_counter(), 0, []
    try:
        for i in range(min(len(videos), workers + 1)):                 # CPU workers start preparing while the model loads
            submit(i)
        det.load_gpu()
        import torch
        torch.cuda.reset_peak_memory_stats()
        for i, video in enumerate(videos):
            submit(i + workers + 1)
            prep = jobs.pop(i).result()
            row = dict(video=video.name, status="ok", n_grids=0, n_fill_grids=0, p_video="", threshold=det.threshold, answer="", t_last_s="",
                       prepare_s=round(prep["secs"], 1), gpu_s=0.0)
            t0 = time.perf_counter()
            answer = None
            try:
                if not prep["ok"]:
                    raise RuntimeError(prep["error"])
                res = det.decide(video.stem, prep["events"], prep["t_last"])
                answer = res["answer"]
                row.update(n_grids=res["n_grids"], n_fill_grids=res["n_fill"], p_video=round(res["p_video"], 4), t_last_s=round(prep["t_last"], 2))
                if res["n_grids"] == 0:
                    row["status"] = "no grids"
            except Exception as e:                                       # a failing video never stops the run: no detection is written
                row["status"] = f"error: {type(e).__name__}: {e}"
                print(f"  ! {video.name}: {row['status']}", flush=True)
            finally:
                shutil.rmtree(Path(work_root) / f"v{i:05d}", ignore_errors=True)
            row["gpu_s"] = round(time.perf_counter() - t0, 1)
            row["answer"] = "" if answer is None else answer
            write_result(results_dir, video.name, answer)
            n_pos += answer is not None
            log.append(row)
            print(f"[{i + 1}/{len(videos)}] {video.name}: {row['status']} | grids {row['n_grids']} | p={row['p_video']} -> "
                  f"{'dumping at ' + str(answer) + ' s' if answer is not None else 'no dumping'} | cpu {row['prepare_s']}s gpu {row['gpu_s']}s", flush=True)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"done: {len(videos)} videos, {n_pos} with a detection, {time.perf_counter() - t_start:.0f}s, GPU peak {peak:.2f} GiB", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(work_root, ignore_errors=True)
        if log:
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
                writer.writeheader()
                writer.writerows(log)
    return 0
