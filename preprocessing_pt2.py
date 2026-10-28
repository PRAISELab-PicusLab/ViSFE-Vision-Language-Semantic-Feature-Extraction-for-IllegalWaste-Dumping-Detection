"""Preprocessing pt. 2: garantisce che ogni tratto del video sia coperto da almeno una griglia.

preprocessing.py crea le griglie solo dove traccia un soggetto in movimento: se il soggetto non viene tracciato
(un lancio istantaneo, un video che inizia a meta' azione, ...) restano buchi, anche video interi senza nessuna
griglia. Questo passo gira SEMPRE subito dopo preprocess_video (non usa le label, funziona identico in produzione):
  - trova i tratti di video non coperti da nessuna griglia evento (testa e coda del video incluse);
  - per ogni tratto lungo almeno GAP_S secondi (2 s) crea griglie sul frame intero (area utile, senza bande nere),
    con le stesse finestre (4 s, passo 2 s, 9 tile) usate da preprocessing.py per la camera mobile.
  - un video senza nessuna griglia e' un unico buco: viene coperto per intero, come una camera mobile.

Le griglie aggiunte si chiamano <video>_f<k>_t<primo>_t<ultimo>.jpg (le griglie evento sono <video>_e<k>_...) e stanno
nella stessa cartella: main.list_strips e vlm_context le leggono senza modifiche. E' idempotente: a ogni esecuzione
le griglie _f esistenti vengono rigenerate. In preprocessing_debug/06_1/<video>/coverage.json resta il motivo.

    python preprocessing_pt2.py                      # tutti i video di events_frames/
    python preprocessing_pt2.py vid0001 vid0002      # solo questi
    python preprocessing_pt2.py --shard 2/4          # worker 2 di 4
"""
import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from decompose_video import FPS_OUT
from preprocessing import STRATEGY, Video, build_parser, render_grid, sample, windows

ROOT = Path(__file__).resolve().parent
FRAMES_ROOT = ROOT / "frames"
STRIPS_ROOT = ROOT / "events_frames"
DEBUG_ROOT = ROOT / "preprocessing_debug"
GAP_S = 2.0  # un tratto scoperto di almeno 2 s (il passo tra finestre) viene riempito: con 4 s un lancio a ~6 s in un buco di 3,5 s (vid0432) restava fuori
_GRID = re.compile(r"_t(\d+)_t(\d+)\.jpg$")
_FILL_GRID = re.compile(r"_f\d+_t\d+_t\d+\.jpg$")


def event_spans(strips_dir, video):
    """[(primo, ultimo)] indici dei frame delle griglie evento (quelle _f gia' presenti non contano)."""
    spans = []
    for path in Path(strips_dir).glob(f"{video}_*.jpg"):
        m = _GRID.search(path.name)
        if m and not _FILL_GRID.search(path.name):
            spans.append((int(m.group(1)), int(m.group(2))))
    return sorted(spans)


def uncovered(n_frames, spans):
    """Intervalli [primo, ultimo] di indici frame non coperti da nessuna griglia (ogni griglia copre primo..ultimo)."""
    covered = [False] * n_frames
    for first, last in spans:
        for i in range(first, min(last, n_frames - 1) + 1):
            covered[i] = True
    gaps, i = [], 0
    while i < n_frames:
        if covered[i]:
            i += 1
            continue
        j = i
        while j + 1 < n_frames and not covered[j + 1]:
            j += 1
        gaps.append((i, j))
        i = j + 1
    return gaps


def fill_coverage(frames_dir, strips_dir, debug_root=None, fps=FPS_OUT, gap_s=GAP_S, jpeg_quality=95):
    """Aggiunge in strips_dir le griglie full-frame per i buchi >= gap_s. Restituisce un dict con l'esito."""
    frames_dir, strips_dir = Path(frames_dir), Path(strips_dir)
    video = frames_dir.name
    for old in strips_dir.glob(f"{video}_f*.jpg"):
        if _FILL_GRID.search(old.name):
            old.unlink()  # rigenerazione idempotente

    n_frames = len(list(frames_dir.glob("frame_*.jpg")))
    spans = event_spans(strips_dir, video)
    min_frames = max(1, round(gap_s * fps))
    gaps = [g for g in uncovered(n_frames, spans) if g[1] - g[0] + 1 >= min_frames]

    names = []
    if gaps:
        args = build_parser().parse_args([str(frames_dir)])
        args.fps = fps
        v = Video(frames_dir, fps, args.analysis_width)
        win, step = max(2, round(args.window * fps)), max(1, round(args.stride * fps))
        wins = sorted({w for first, last in gaps for w in windows(first, last, v.T, win, step)})
        for k, (a, b) in enumerate(wins):
            idxs = sample(a, b, args.tiles)
            name = f"{video}_f{k:03d}_t{idxs[0]}_t{idxs[-1]}.jpg"
            render_grid(v, idxs, v.content, args).convert("RGB").save(strips_dir / name, quality=jpeg_quality)
            names.append(name)

    result = {"video": video, "gap_threshold_s": gap_s, "n_frames": n_frames, "n_event_grids": len(spans),
              "reason": "no_grids" if not spans else ("gaps" if gaps else "covered"),
              "gaps_filled": [{"start_s": f / fps, "end_s": (l + 1) / fps, "length_s": (l - f + 1) / fps} for f, l in gaps],
              "fill_grids": names}
    if debug_root is not None:
        out = Path(debug_root) / STRATEGY / video
        out.mkdir(parents=True, exist_ok=True)
        (out / "coverage.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("videos", nargs="*", help="default: tutti i video di events_frames/")
    parser.add_argument("--shard", default=None, help="i/n: elabora solo i video con indice % n == i-1")
    parser.add_argument("--gap", type=float, default=GAP_S, help=f"secondi minimi di un buco da riempire (default {GAP_S:g})")
    args = parser.parse_args()
    names = args.videos or sorted(p.name for p in STRIPS_ROOT.iterdir() if p.is_dir() and not p.name.endswith(".tmp"))
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        names = [v for k, v in enumerate(names) if k % n == i - 1]
    t0 = time.perf_counter()
    added = filled = 0
    for video in tqdm(names, desc="copertura", unit="video"):
        result = fill_coverage(FRAMES_ROOT / video, STRIPS_ROOT / video, debug_root=DEBUG_ROOT, gap_s=args.gap)
        if result["fill_grids"]:
            filled += 1
            added += len(result["fill_grids"])
            tqdm.write(f"{video}: {result['reason']}, +{len(result['fill_grids'])} griglie "
                       f"({len(result['gaps_filled'])} buchi)")
    print(f"Fatto in {time.perf_counter() - t0:.0f}s. Video riempiti: {filled}/{len(names)}, griglie aggiunte: {added}")


if __name__ == "__main__":
    main()
