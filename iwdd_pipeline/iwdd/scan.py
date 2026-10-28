"""Scansione dei video (metadati reali, firma del fondo) e gruppi di camera per la cross-validation.

Per ogni video: dimensioni, pts reali (CAP_PROP_POS_MSEC), fps misurato, luminosita', firma del fondo mediano (32x18 grigi).
I gruppi (componenti connesse: stesso rapporto d'aspetto e correlazione della firma >= soglia) servono a tenere nello stesso
fold i video della stessa telecamera.
"""
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .paths import DATASET, Run

log = logging.getLogger("iwdd.scan")


def scan_video(path, sample_every=0.5, small_width=160, sig_size=(32, 18)):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire il video: {path}")
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n, first, last, nxt = 0, None, 0.0, 0.0
    smalls = []
    while cap.grab():
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        n += 1
        first = t if first is None else first
        last = t
        if t >= nxt:
            ok, frame = cap.retrieve()
            if ok:
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                s = small_width / g.shape[1]
                smalls.append(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
            nxt = t + sample_every
    cap.release()
    if not smalls:
        raise RuntimeError(f"Nessun frame leggibile: {path}")
    sig = np.median(np.stack([cv2.resize(s, sig_size, interpolation=cv2.INTER_AREA) for s in smalls]).astype(np.float32), axis=0)
    return {
        "width": w0, "height": h0, "n_frames": n, "t0": float(first), "t_last": float(last),
        "fps_mean": float((n - 1) / (last - first)) if n > 1 and last > first else None,
        "brightness": float(np.mean([s.mean() for s in smalls])),
        "bg_signature": np.round(sig, 1).flatten().tolist(),
    }


def _scan_job(args):
    name, split, vid, kw = args
    run = Run(name)
    out = run.meta / f"{vid}.json"
    if out.exists():
        return vid, "gia' presente"
    t0 = time.perf_counter()
    try:
        m = scan_video(DATASET / split / "videos" / f"{vid}.mp4", **kw)
    except Exception as e:  # un video difettoso non ferma il lotto
        return vid, f"ERRORE {type(e).__name__}: {e}"
    m.update(video=vid, split=split, scan_s=round(time.perf_counter() - t0, 1))
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(m), encoding="utf-8")
    tmp.replace(out)
    return vid, f"ok {m['scan_s']}s"


def scan_all(run, videos, cfg, workers=4):
    kw = dict(sample_every=cfg["scan"]["sample_every_s"], small_width=cfg["scan"]["small_width"],
              sig_size=tuple(cfg["scan"]["sig_size"]))
    jobs = [(run.name, v.split, v.vid, kw) for v in videos]
    errors = 0
    with ProcessPoolExecutor(workers) as ex:
        for i, (vid, msg) in enumerate(ex.map(_scan_job, jobs), 1):
            errors += msg.startswith("ERRORE")
            if msg.startswith("ERRORE") or i % 25 == 0 or i == len(jobs):
                log.info("[%d/%d] %s: %s", i, len(jobs), vid, msg)
    log.info("scan completata: %d video, %d errori", len(jobs), errors)
    return errors


def load_meta(run, vids):
    out = {}
    for v in vids:
        p = run.meta / f"{v}.json"
        if p.exists():
            out[v] = json.loads(p.read_text(encoding="utf-8"))
    return out


def compute_groups(meta, sim=0.95):
    """{video: id gruppo}. Stesso rapporto d'aspetto (2 decimali) e correlazione della firma >= sim -> stesso gruppo."""
    names = sorted(meta)
    if not names:
        return {}
    X = np.array([meta[n]["bg_signature"] for n in names], dtype=np.float32)
    X = X - X.mean(axis=1, keepdims=True)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    C = X @ X.T
    ar = [round(meta[n]["width"] / meta[n]["height"], 2) for n in names]
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if ar[i] == ar[j] and C[i, j] >= sim:
                parent[find(i)] = find(j)
    roots = {}
    return {n: roots.setdefault(find(i), len(roots)) for i, n in enumerate(names)}


def save_groups(run, videos, cfg):
    meta = load_meta(run, [v.vid for v in videos])
    groups = compute_groups(meta, cfg["scan"]["group_sim"])
    # scrittura atomica: con l'addestramento in parallelo piu' job ricalcolano (identici) i gruppi nello stesso momento
    out = run.meta / "groups.json"
    tmp = out.with_name(f"groups.json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"sim": cfg["scan"]["group_sim"], "group": groups}, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    n_g = len(set(groups.values()))
    log.info("gruppi di camera: %d gruppi su %d video (%d video senza scansione)", n_g, len(groups), len(videos) - len(groups))
    return groups


def load_groups(run):
    p = run.meta / "groups.json"
    return json.loads(p.read_text(encoding="utf-8"))["group"] if p.exists() else {}
