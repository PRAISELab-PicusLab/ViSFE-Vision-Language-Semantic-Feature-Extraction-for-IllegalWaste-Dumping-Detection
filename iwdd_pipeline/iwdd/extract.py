"""Estrazione: per ogni immagine evento, le risposte (logit grezzi) di Visual-JEV a tutte le domande. Resumabile e a shard.

Uscite in <run>/features/:
  <video>.parquet          una riga per (griglia, domanda): video, win, qid, n_opt, l0..l3 (logit grezzi, NaN oltre n_opt)
  <video>.windows.parquet  una riga per griglia: video, win, t_start, t_center, t_end, image, image_tokens, t_win, kind
                           (kind: 'event' = griglia del preprocessing, 'fill' = griglia di copertura di preprocessing_pt2.py)
  _manifest.json           impronta di domande/contesto/risoluzione: impedisce di mescolare feature incompatibili
  timing_shardK.jsonl      tempi e memoria
Un video e' completo quando esiste <video>.parquet (scritto per ultimo, in modo atomico).
"""
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import shard
from .events import grid_kind, images_dir, list_event_images
from .questions import context_text, questions_hash

log = logging.getLogger("iwdd.extract")
MAXOPT = 4
LONG_COLS = {"video": "str", "win": "int64", "qid": "str", "n_opt": "int64", "l0": "float32", "l1": "float32", "l2": "float32", "l3": "float32"}
WIN_COLS = {"video": "str", "win": "int64", "t_start": "float64", "t_center": "float64", "t_end": "float64", "image": "str",
            "image_tokens": "int64", "t_win": "float64", "kind": "str"}


def _empty(cols):
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()})


def _atomic_parquet(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_text(path, text):
    """Scrittura atomica (file temporaneo per processo + rename): piu' task in parallelo scrivono gli stessi file del run."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def check_manifest(run, questions, context_mode, max_pixels, force=False, fill_gaps_s=None):
    mf = run.features / "_manifest.json"
    # QUANT_BITS: variabile letta da main.py del progetto (8 = default validato, 4 = fallback NF4): cambia le risposte del modello
    cur = {"questions_hash": questions_hash(questions, context_mode, max_pixels), "context_mode": context_mode,
           "max_pixels": max_pixels, "question_ids": [q.id for q in questions], "quant_bits": os.environ.get("QUANT_BITS", "8"),
           "fill_gaps_s": fill_gaps_s}
    if mf.exists():
        old = json.loads(mf.read_text(encoding="utf-8"))
        if old.get("quant_bits", "8") != cur["quant_bits"] and not force:
            raise RuntimeError(f"Le feature di {run.features} sono state estratte con QUANT_BITS={old.get('quant_bits', '8')} e ora "
                               f"QUANT_BITS={cur['quant_bits']}: le risposte del modello cambiano. Usa lo stesso valore, un altro --run, oppure --force.")
        if old.get("fill_gaps_s") != cur["fill_gaps_s"] and not force:     # manifest vecchi senza chiave = senza copertura
            raise RuntimeError(f"Le feature di {run.features} sono state estratte con fill_gaps_s={old.get('fill_gaps_s')} e ora "
                               f"fill_gaps_s={cur['fill_gaps_s']}: le griglie sono diverse. Usa lo stesso valore, un altro --run, oppure --force.")
        if old["questions_hash"] != cur["questions_hash"] and not force:
            raise RuntimeError(
                f"Le feature in {run.features} sono state estratte con un set di domande/contesto/risoluzione DIVERSO "
                f"({old['questions_hash']} contro {cur['questions_hash']}). Usa un altro --run, oppure --force per sovrascrivere.")
    _atomic_text(mf, json.dumps(cur, indent=1))
    # copia leggibile delle domande esatte usate in questo run (audit)
    _atomic_text(run.root / "questions_used.json", json.dumps(
        [{"id": q.id, "kind": q.kind, "text": q.en, "options": list(q.options)} for q in questions], indent=1, ensure_ascii=False))
    try:
        from .questions import render_prompts
        _atomic_text(run.root / "prompts_used.txt",
                     "\n\n".join(f"##### {qid}\n{txt}" for qid, txt in render_prompts(questions, context_mode)))
    except FileNotFoundError:
        pass
    return cur


def _gpu_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _gpu_gib(peak=False):
    """GiB allocati sulla GPU (0 se non c'e' GPU: serve ai test senza GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return (torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()) / 1024 ** 3
    except Exception:
        pass
    return 0.0


def video_done(run, vid):
    return (run.features / f"{vid}.parquet").exists()


def extract(run, videos, questions, cfg, shard_idx=0, num_shards=1, chunk=None, max_pixels=None, context_mode=None,
            max_minutes=0, force=False, reuse_dirs=(), limit_videos=None, wait_for=None, poll_s=3.0):
    """Ritorna dict(done, remaining, errors, missing). remaining > 0 solo se si e' fermata per il budget di tempo.

    wait_for: modalita' "in parallelo a prepare". Il modello si carica subito; poi si estraggono i video man mano che le loro
    immagini evento compaiono (prepare le scrive in <video>.tmp e rinomina: se la cartella esiste e' completa). Quando il file
    wait_for esiste (prepare ha finito) e non c'e' piu' nulla di pronto, ci si ferma: i video rimasti senza immagini sono 'missing'."""
    jev = cfg["jev"]
    chunk = chunk or jev["chunk"]
    max_pixels = max_pixels or jev["max_pixels"]
    context_mode = context_mode or jev["context_mode"]
    fps = cfg["events"]["fps"]
    check_manifest(run, questions, context_mode, max_pixels, force, cfg["events"].get("fill_gaps_s"))

    # shard bilanciati sul numero di immagini di ogni video
    dirs = {v.vid: images_dir(run, v.vid, reuse_dirs) for v in videos}
    weights = {vid: (len(list_event_images(d)) if d else 1) for vid, d in dirs.items()}
    mine = shard(videos, shard_idx, num_shards, weights)
    missing = [v.vid for v in mine if dirs[v.vid] is None]
    todo = [v for v in mine if (wait_for or dirs[v.vid] is not None) and (force or not video_done(run, v.vid))]
    if limit_videos:
        todo = todo[:limit_videos]
    log.info("shard/parte %d (di %d shard): %d video assegnati, %d da estrarre, %d ancora senza immagini evento%s",
             shard_idx, num_shards, len(mine), len(todo), len(missing),
             f" (li aspetto: prepare in corso, fine segnalata da {wait_for})" if wait_for else " (esegui prima 'prepare')")
    if not todo:
        return dict(done=0, remaining=0, errors=0, missing=len(missing))

    from PIL import Image
    from .jev import build_scorer
    scorer = build_scorer(max_pixels)
    log.info("modello caricato: %.2f GiB su GPU", _gpu_gib())
    qlist = [q.jev_item() for q in questions]
    t_start, done, errors = time.perf_counter(), 0, 0
    per_img, t_wait = [], 0.0
    pending, n_v = list(todo), 0
    while pending:
        elapsed = (time.perf_counter() - t_start) / 60.0
        if max_minutes and elapsed > max_minutes:
            log.warning("budget di tempo (%.0f min) raggiunto: mi fermo con %d video ancora da fare", max_minutes, len(pending))
            return dict(done=done, remaining=len(pending), errors=errors, missing=len(missing))
        ready = [v for v in pending if images_dir(run, v.vid, reuse_dirs) is not None]
        if not ready:
            if wait_for and not Path(wait_for).exists():
                time.sleep(poll_s)                              # prepare sta ancora lavorando: aspetto il prossimo video
                t_wait += poll_s
                continue
            ready = [v for v in pending if images_dir(run, v.vid, reuse_dirs) is not None]   # prepare finito: ultimo controllo
            if not ready:
                break
        v = ready[0]
        pending.remove(v)
        n_v += 1
        try:
            imgs = list_event_images(images_dir(run, v.vid, reuse_dirs))
            _gpu_reset()
            rows, wrows = [], []
            t0 = time.perf_counter()
            for k, (a, b, path) in enumerate(imgs):
                with Image.open(path) as src:
                    image = scorer.jev_main.crop_margins(src.convert("RGB"))
                dt = (b - a) / fps / 8.0 if b > a else 0.5
                lg, info = scorer.score_raw(image, qlist, context=context_text(context_mode, dt), chunk=chunk)
                ts, te = a / fps, b / fps
                wrows.append((v.vid, k, ts, (ts + te) / 2, te, path.name, info["image_tokens"],
                              info["t_prepare"] + info["t_prefill"] + info["t_suffix"], grid_kind(path.name)))
                for qi, q in enumerate(questions):
                    l = np.full(MAXOPT, np.nan, dtype=np.float32)
                    l[:q.n_options] = lg[qi, :q.n_options]
                    rows.append((v.vid, k, q.id, q.n_options, *l))
            t_inf = time.perf_counter() - t0
            df = pd.DataFrame(rows, columns=list(LONG_COLS)) if rows else _empty(LONG_COLS)
            dw = pd.DataFrame(wrows, columns=list(WIN_COLS)) if wrows else _empty(WIN_COLS)
            _atomic_parquet(dw.astype(WIN_COLS), run.features / f"{v.vid}.windows.parquet")
            _atomic_parquet(df.astype(LONG_COLS), run.features / f"{v.vid}.parquet")     # per ultimo: marca il video come completo
            done += 1
            per_img.append((len(imgs), t_inf))
            rec = {"video": v.vid, "n_images": len(imgs), "n_questions": len(questions), "t_infer_s": round(t_inf, 2),
                   "s_per_image": round(t_inf / max(1, len(imgs)), 3), "peak_alloc_gib": round(_gpu_gib(peak=True), 3),
                   "chunk": chunk, "max_pixels": max_pixels, "context_mode": context_mode}
            with open(run.features / f"timing_shard{shard_idx}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            tot_i = sum(n for n, _ in per_img)
            log.info("[%d/%d] %s: %d immagini in %.1fs (%.2f s/img, picco %.2f GiB; media %.2f s/img%s)", n_v, len(todo), v.vid, len(imgs),
                     t_inf, rec["s_per_image"], rec["peak_alloc_gib"], sum(t for _, t in per_img) / max(1, tot_i),
                     f"; GPU in attesa di prepare finora {t_wait:.0f}s" if wait_for else "")
        except Exception as e:
            errors += 1
            log.exception("ERRORE su %s", v.vid)
            with open(run.logs / f"extract_errors_shard{shard_idx}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"video": v.vid, "error": f"{type(e).__name__}: {e}"}) + "\n")
    missing = [v.vid for v in mine if images_dir(run, v.vid, reuse_dirs) is None]
    if missing:
        log.warning("%d video senza immagini evento (prepare fallito?): %s", len(missing), missing[:10])
    return dict(done=done, remaining=0, errors=errors, missing=len(missing))
