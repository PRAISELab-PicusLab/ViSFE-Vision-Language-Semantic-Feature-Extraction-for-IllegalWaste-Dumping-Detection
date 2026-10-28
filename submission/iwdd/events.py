"""Immagini evento: il preprocessing esistente del progetto (preprocessing.py, strategia 06_1: camera fissa/mobile, tracking,
zoom sul soggetto, griglia 3x3 da 9 frame) applicato ai video, SENZA modificarlo.

Per ogni video: decompose_video (4 fps) -> preprocess_video -> <run>/events/<video>/<video>_eNNN_tA_tB.jpg
            -> fill_coverage (preprocessing_pt2.py) -> griglie a frame intero <video>_fNNN_tA_tB.jpg sui tratti scoperti
Gli indici A,B sono indici di frame a 4 fps: t = indice / 4 (stessa convenzione della pipeline originale).
Tutto si scrive in <video>.prep e si rinomina alla fine: se events/<video> esiste e' completa, griglie di copertura comprese
(extract --wait-for si basa su questo). Il riempimento si regola con events.fill_gaps_s in config (null = disattivato).
Si possono riusare cartelle di immagini gia' prodotte (es. events_frames/) con --reuse-events: sola lettura.
"""
import json
import logging
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .paths import DATASET, Run, ensure_project_imports

log = logging.getLogger("iwdd.events")
NAME_RE = re.compile(r"_t(\d+)_t(\d+)\.jpg$")
FILL_RE = re.compile(r"_f\d+_t\d+_t\d+\.jpg$")        # griglie di copertura (preprocessing_pt2.py)


def grid_kind(name):
    """'fill' per le griglie di copertura a frame intero, 'event' per quelle del preprocessing (zoom sul soggetto)."""
    return "fill" if FILL_RE.search(name) else "event"


def images_dir(run, vid, reuse_dirs=()):
    """Cartella con le immagini evento di un video: prima quella del run, poi le cartelle da riusare. None se manca."""
    for d in [run.root / "events" / vid] + [Path(r) / vid for r in reuse_dirs]:
        if d.is_dir():
            return d
    return None


def list_event_images(directory):
    """[(indice_inizio, indice_fine, percorso)] ordinati per tempo di fine."""
    out = []
    for p in Path(directory).glob("*.jpg"):
        m = NAME_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(out, key=lambda x: (x[1], x[0], x[2].name))


def _prepare_job(args):
    name, split, vid, reuse_dirs, fps, quality, fill_gaps_s = args
    run = Run(name)
    if images_dir(run, vid, reuse_dirs) is not None:
        return vid, "gia' pronto", 0.0
    ensure_project_imports()
    from decompose_video import decompose_video
    from preprocessing import preprocess_video
    scratch = run.scratch()
    path = DATASET / split / "videos" / f"{vid}.mp4"
    prep = run.events / f"{vid}.prep"     # cartella di lavoro: diventa events/<vid> solo quando e' completa
    t0 = time.perf_counter()
    last_err = None
    for attempt in range(3):        # Windows: il rename della cartella puo' fallire per un attimo (antivirus)
        try:
            shutil.rmtree(prep, ignore_errors=True)
            decompose_video(path, fps_out=fps, output_root=scratch)
            imgs = preprocess_video(scratch / vid, prep, debug_root=None, fps=fps, jpeg_quality=quality)
            msg = f"{len(imgs)} immagini"
            if fill_gaps_s is not None:
                from preprocessing_pt2 import fill_coverage
                cov = fill_coverage(scratch / vid, prep, debug_root=None, fps=fps, gap_s=fill_gaps_s, jpeg_quality=quality)
                (prep / "coverage.json").write_text(json.dumps(cov, indent=1), encoding="utf-8")
                msg += f" + {len(cov['fill_grids'])} di copertura ({cov['reason']})"
            prep.rename(run.events / vid)
            return vid, msg, time.perf_counter() - t0
        except PermissionError as e:
            last_err = e
            time.sleep(2 + 3 * attempt)
            shutil.rmtree(scratch / (vid + ".tmp"), ignore_errors=True)
        except Exception as e:
            return vid, f"ERRORE {type(e).__name__}: {e}", time.perf_counter() - t0
        finally:
            shutil.rmtree(scratch / vid, ignore_errors=True)
    return vid, f"ERRORE PermissionError: {last_err}", time.perf_counter() - t0


def prepare_all(run, videos, cfg, workers=4, reuse_dirs=()):
    # dal video piu' lungo al piu' corto: i lenti partono subito invece di restare per ultimi (tempo totale minore con N processi)
    order = sorted(videos, key=lambda v: (-v.duration, v.vid))
    fill = cfg["events"].get("fill_gaps_s")
    jobs = [(run.name, v.split, v.vid, tuple(map(str, reuse_dirs)), cfg["events"]["fps"], cfg["events"]["jpeg_quality"], fill)
            for v in order]
    errors, t_all = [], time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        for i, (vid, msg, dt) in enumerate(ex.map(_prepare_job, jobs), 1):
            if msg.startswith("ERRORE"):
                errors.append((vid, msg))
            if msg.startswith("ERRORE") or i % 10 == 0 or i == len(jobs):
                log.info("[%d/%d] %s: %s (%.0fs)", i, len(jobs), vid, msg, dt)
    if errors:
        with open(run.logs / "prepare_errors.jsonl", "a", encoding="utf-8") as f:
            for vid, msg in errors:
                f.write(f'{{"video": "{vid}", "error": {msg!r}}}\n')
    log.info("preparazione completata in %.0f s: %d video, %d errori", time.perf_counter() - t_all, len(jobs), len(errors))
    return len(errors)
