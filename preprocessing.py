"""
Strategia 06_1 - identica alla 06, ma SENZA la striscia di contesto in cima alle immagini
(miniatura con il riquadro rosso): le griglie occupano tutta l'immagine e i riquadri restano più grandi.
Originale (06) - come la 05 (camera fissa o no), ma:
  - niente strobo: solo griglie
  - tracciamento più robusto quando i soggetti si incrociano
    (posizione prevista + dimensione + colore, assegnazione ottima invece che "il più vicino")
  - soggetti vicini nello stesso momento -> un'unica immagine ("evento") invece di una per soggetto
  - PRIORITÀ per ordinare gli eventi (non scarta mai nulla, tutte le immagini vengono generate):
        sosta   = per quanto il soggetto resta fermo nella finestra
        traccia = nella zona inquadrata compare qualcosa di piccolo che prima non c'era e dopo resta fermo
                  (copre anche chi lascia cadere qualcosa senza fermarsi)
    Gli indizi NON vengono disegnati sulle immagini, per non condizionare il VLM: stanno solo nei JSON.
  - analisi a risoluzione più alta (960 px) per non perdere oggetti piccoli

Output in output/06_1/<video>/:
  camera.json      decisione fissa/mobile con le misure
  tracks.json      soggetti trovati (solo camera fissa)
  overview.png     scena con i percorsi dei soggetti (solo camera fissa)
  grid_crop/ o grid_full/   immagini per il VLM + index.json (tempi di immagini e riquadri, indizi di priorità)
  priority.json    eventi ordinati per priorità (solo camera fissa)

Esempi:
    python strategies/06_events_priority/run.py frames
    python strategies/06_events_priority/run.py frames/vid0004
"""

import argparse
import json
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
STRATEGY = "06_1"  # nome fisso: il file non sta piu in strategies/<nome>/run.py


# ================================================================ utilità

def list_frames(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def video_fps(folder: Path, default: float) -> float:
    meta = folder / "meta.json"
    if meta.exists():
        return float(json.loads(meta.read_text(encoding="utf-8")).get("fps", default))
    return default


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def label(img: Image.Image, text: str, size: int, pos=(0, 0), color=(255, 255, 0)) -> None:
    d = ImageDraw.Draw(img)
    f = font(size)
    x0, y0, x1, y1 = d.textbbox(pos, text, font=f)
    d.rectangle([x0, y0, x1 + 2 * (size // 3), y1 + size // 3], fill=(0, 0, 0))
    d.text((pos[0] + size // 3, pos[1]), text, fill=color, font=f)


def windows(first: int, last: int, T: int, win: int, step: int) -> list[tuple[int, int]]:
    """Finestre [a, b) di lunghezza win che coprono [first, last]; se il tratto è corto, una finestra centrata."""
    win = min(win, T)
    if last - first + 1 <= win:
        c = (first + last) // 2
        a = int(np.clip(c - win // 2, 0, T - win))
        return [(a, a + win)]
    out = [(a, a + win) for a in range(first, last - win + 2, step)]
    if out[-1][1] <= last:
        out.append((last + 1 - win, last + 1))
    return out


def sample(a: int, b: int, n: int) -> list[int]:
    return sorted(set(np.linspace(a, b - 1, min(n, b - a)).round().astype(int).tolist()))


def iou(p, q) -> float:
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / max(area(p) + area(q) - inter, 1)


# ================================================================ caricamento

class Video:
    def __init__(self, folder: Path, fps: float, analysis_width: int):
        self.folder = folder
        self.files = list_frames(folder)
        self.T = len(self.files)
        self.fps = fps
        first = cv2.imread(str(self.files[0]))
        self.H, self.W = first.shape[:2]
        self.s = min(1.0, analysis_width / self.W)  # scala analisi
        self.small = np.stack([cv2.resize(cv2.imread(str(f)), None, fx=self.s, fy=self.s,
                                          interpolation=cv2.INTER_AREA) for f in self.files])
        self.content = self._content_box()  # (x1,y1,x2,y2) a piena ris., senza bande nere

    def _content_box(self) -> tuple[int, int, int, int]:
        mx = self.small.max(axis=3).max(axis=0).astype(np.float32)  # una banda nera resta sempre scura
        rows = np.nonzero(mx.mean(axis=1) > 25)[0]
        cols = np.nonzero(mx.mean(axis=0) > 25)[0]
        if not len(rows) or not len(cols):
            return 0, 0, self.W, self.H
        return (int(cols[0] / self.s), int(rows[0] / self.s),
                min(self.W, int((cols[-1] + 1) / self.s)), min(self.H, int((rows[-1] + 1) / self.s)))

    def to_small(self, box) -> tuple[int, int, int, int]:
        return tuple(int(round(c * self.s)) for c in box)

    @lru_cache(maxsize=96)
    def full(self, i: int) -> Image.Image:
        return Image.open(self.files[i]).convert("RGB")


# ================================================================ fase 1: camera fissa?

def camera_check(v: Video, args) -> dict:
    x1, y1, x2, y2 = v.to_small(v.content)
    gray = np.stack([cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f[y1:y2, x1:x2], None, fx=0.5, fy=0.5),
                                                   cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in v.small])
    T = len(gray)
    k = max(1, min(round(v.fps), T - 1))  # confronto a ~1 secondo di distanza
    hann = cv2.createHanningWindow(gray.shape[2:0:-1], cv2.CV_32F)
    shifts, statics = [], []
    for t in range(0, T - k, max(1, k // 2)):
        a, b = gray[t], gray[t + k]
        (dx, dy), _ = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), hann)
        shifts.append(float(np.hypot(dx, dy) / a.shape[1]))
        statics.append(float((cv2.absdiff(a, b) < 12).mean()))
    static = float(np.median(statics)) if statics else 1.0
    # mediana: un veicolo grande che passa può ingannare qualche confronto, ma non la maggior parte
    shift = float(np.median(shifts)) if shifts else 0.0
    fixed = static >= args.static_min and shift <= args.shift_max
    return {"fixed": bool(fixed), "static_ratio": round(static, 3), "shift_median": round(shift, 4),
            "content_box": list(v.content), "resolution": [v.W, v.H]}


# ================================================================ camera fissa: primo piano e soggetti

def foreground(v: Video, args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Maschere dei soggetti in movimento (fg: pezzi uniti, per il tracciamento; fg_raw: senza unione,
    così un oggetto piccolo vicino a un soggetto resta separato) e soglia per pixel (coordinate analisi)."""
    st = v.small
    idx = np.linspace(0, v.T - 1, min(60, v.T)).round().astype(int)
    bg = np.median(st[idx], axis=0).astype(np.uint8)
    diffs = np.stack([cv2.absdiff(f, bg).max(axis=2) for f in st])
    noise = np.median(diffs, axis=0)  # zone che "ballano" (erba, acqua) -> soglia più alta
    thr = args.threshold + args.noise_k * noise
    w = bg.shape[1]
    kc = max(3, round(args.close_frac * w)) | 1
    k_open = np.ones((3, 3), np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    x1, y1, x2, y2 = v.to_small(v.content)
    fg = np.zeros(diffs.shape, dtype=np.uint8)
    fg_raw = np.zeros(diffs.shape, dtype=np.uint8)
    for t in range(v.T):
        m = (diffs[t] > thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
        fg_raw[t, y1:y2, x1:x2] = m[y1:y2, x1:x2]
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)  # unisce i pezzi dello stesso soggetto
        fg[t, y1:y2, x1:x2] = m[y1:y2, x1:x2]
    return fg, fg_raw, thr, bg


def track(v: Video, fg: np.ndarray, args) -> list[dict]:
    """Tracciamento: posizione prevista (velocità costante) + dimensione + colore, assegnazione ottima."""
    h, w = fg.shape[1:]
    min_area = args.min_area_frac * w * h
    max_gap = max(1, round(args.max_gap * v.fps))
    tracks, active = [], []
    for t in range(v.T):
        n, lab, stats, cents = cv2.connectedComponentsWithStats(fg[t], connectivity=8)
        blobs = []
        for i in range(1, n):
            if stats[i, 4] < min_area:
                continue
            col = np.array(cv2.mean(v.small[t], mask=(lab == i).astype(np.uint8))[:3])
            blobs.append({"box": stats[i, :4].astype(float), "c": cents[i], "area": float(stats[i, 4]), "col": col})

        assigned = {}
        if active and blobs:
            C = np.full((len(active), len(blobs)), 1e6)
            for r, tr in enumerate(active):
                last_c = np.array(tr["cents"][-1])
                if len(tr["frames"]) >= 2:
                    vel = (last_c - np.array(tr["cents"][-2])) / (tr["frames"][-1] - tr["frames"][-2])
                else:
                    vel = np.zeros(2)
                pred = last_c + vel * (t - tr["frames"][-1])
                size = max(tr["boxes"][-1][2:])
                for j, b in enumerate(blobs):
                    d = np.hypot(*(b["c"] - pred)) / size          # spostamento in "dimensioni del soggetto"
                    ratio = b["area"] / tr["area"]
                    if d > args.max_jump or not (1 / args.max_size_ratio <= ratio <= args.max_size_ratio):
                        continue                                    # troppo lontano o dimensione troppo diversa
                    C[r, j] = d + args.color_w * np.linalg.norm(b["col"] - tr["col"]) / 100
            for r, j in zip(*linear_sum_assignment(C)):
                if C[r, j] < 1e6:
                    assigned[j] = active[r]

        for j, b in enumerate(blobs):
            tr = assigned.get(j)
            if tr is None:
                tr = {"frames": [], "boxes": [], "cents": [], "area": b["area"], "col": b["col"]}
                tracks.append(tr)
                active.append(tr)
            tr["frames"].append(t)
            tr["boxes"].append(b["box"])
            tr["cents"].append(b["c"])
            tr["area"] = 0.7 * tr["area"] + 0.3 * b["area"]
            tr["col"] = 0.7 * tr["col"] + 0.3 * b["col"]
        active = [tr for tr in active if t - tr["frames"][-1] <= max_gap]

    out = []
    min_len = max(2, round(args.min_track * v.fps))
    diag = np.hypot(w, h)
    for tr in tracks:
        if len(tr["frames"]) < min_len:
            continue
        c = np.array(tr["cents"])
        f = np.array(tr["frames"])
        speed = np.zeros(len(c))
        if len(c) > 1:
            speed[1:] = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.diff(f) / diag * v.fps
            speed[0] = speed[1]
        boxes = np.array(tr["boxes"])
        out.append({
            "frames": tr["frames"], "boxes": (boxes / v.s).round().astype(int).tolist(),  # piena ris. x,y,w,h
            "cents": (c / v.s).round().astype(int).tolist(), "speed": speed.round(4).tolist(),
            "first": tr["frames"][0], "last": tr["frames"][-1],
            "start_s": round(tr["frames"][0] / v.fps, 2), "end_s": round(tr["frames"][-1] / v.fps, 2),
            "max_size_frac": round(float((boxes[:, 2] * boxes[:, 3]).max() / (w * h)), 4),
            "stopped_s": round(float((speed < args.stop_speed).sum() / v.fps), 2),
        })
    out.sort(key=lambda r: r["first"])
    for k, r in enumerate(out):
        r["id"] = k
    return out


# ================================================================ eventi: finestre per soggetto, poi unione

def fit_box(raw, v: Video, args) -> tuple[int, int, int, int]:
    """Da un rettangolo 'grezzo' (x1,y1,x2,y2) alla zona da ingrandire: margine, dimensione minima, rapporto video."""
    x1, y1, x2, y2 = raw
    cx1, cy1, cx2, cy2 = v.content
    CW, CH = cx2 - cx1, cy2 - cy1
    pad = args.crop_pad * max(x2 - x1, y2 - y1)
    w = max(x2 - x1 + 2 * pad, CW * args.crop_min)
    h = max(y2 - y1 + 2 * pad, CH * args.crop_min)
    ar = CW / CH
    w, h = (w, w / ar) if w / h > ar else (h * ar, h)
    w, h = min(w, CW), min(h, CH)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1 = int(np.clip(cx - w / 2, cx1, cx2 - w))
    Y1 = int(np.clip(cy - h / 2, cy1, cy2 - h))
    return X1, Y1, int(X1 + w), int(Y1 + h)


def build_events(tracks: list[dict], v: Video, args) -> list[dict]:
    win = max(2, round(args.window * v.fps))
    step = max(1, round(args.stride * v.fps))
    events = []
    for tr in tracks:
        for a, b in windows(tr["first"], tr["last"], v.T, win, step):
            sel = [k for k, f in enumerate(tr["frames"]) if a <= f < b] or list(range(len(tr["frames"])))
            bx = np.array([tr["boxes"][k] for k in sel])
            raw = [int(bx[:, 0].min()), int(bx[:, 1].min()),
                   int((bx[:, 0] + bx[:, 2]).max()), int((bx[:, 1] + bx[:, 3]).max())]
            stop = sum(1 for k in sel if tr["speed"][k] < args.stop_speed) / v.fps
            events.append({"tracks": [tr["id"]], "a": a, "b": b, "raw": raw, "stop_s": stop})

    # unisci eventi che si sovrappongono nel tempo e nello spazio (stessa zona, stesso momento)
    content_area = (v.content[2] - v.content[0]) * (v.content[3] - v.content[1])
    merged = []
    for e in sorted(events, key=lambda e: e["a"]):
        for m in merged:
            overlap = min(m["b"], e["b"]) - max(m["a"], e["a"])
            if overlap < 0.5 * min(m["b"] - m["a"], e["b"] - e["a"]):
                continue
            if max(m["b"], e["b"]) - min(m["a"], e["a"]) > 1.5 * win:
                continue
            if iou(fit_box(m["raw"], v, args), fit_box(e["raw"], v, args)) < args.merge_iou:
                continue
            u = [min(m["raw"][0], e["raw"][0]), min(m["raw"][1], e["raw"][1]),
                 max(m["raw"][2], e["raw"][2]), max(m["raw"][3], e["raw"][3])]
            fb = fit_box(u, v, args)
            if (fb[2] - fb[0]) * (fb[3] - fb[1]) > args.merge_max_area * content_area:
                continue  # l'unione diventerebbe troppo grande: meglio due immagini
            m.update(a=min(m["a"], e["a"]), b=max(m["b"], e["b"]), raw=u, stop_s=max(m["stop_s"], e["stop_s"]),
                     tracks=sorted(set(m["tracks"] + e["tracks"])))
            break
        else:
            merged.append(dict(e))
    for e in merged:
        e["box"] = fit_box(e["raw"], v, args)
    return merged


def subject_mask(fg: np.ndarray, frames: range, args) -> np.ndarray:
    """Unione (dilatata) dei soggetti in movimento nei frame dati: lì la 'traccia' non conta."""
    h, w = fg.shape[1:]
    out = np.zeros((h, w), np.uint8)
    for t in frames:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg[t], connectivity=8)
        big = np.nonzero(stats[:, 4] >= args.min_area_frac * w * h)[0]
        big = big[big > 0]
        if len(big):
            out |= np.isin(lab, big).astype(np.uint8)
    return cv2.dilate(out, np.ones((3, 3), np.uint8), iterations=2)


def left_behind(v: Video, fg: np.ndarray, thr: np.ndarray, a: int, b: int, box, args) -> dict:
    """Qualcosa di piccolo compare nella zona tra prima e dopo la finestra e resta fermo?
    Calcolato a PIENA risoluzione dentro la zona ritagliata (gli oggetti piccoli a risoluzione ridotta spariscono).
      - mediane di ~1.5s attorno all'inizio (prima) e alla fine (dopo) della finestra
      - conta solo dove prima e dopo l'immagine è stabile (chi si muove non conta)
      - il 'dopo' deve essere diverso dallo sfondo di tutto il video (una zona che torna libera non conta)
      - si escludono i soggetti presenti 'dopo' (un'auto ferma non è una traccia) e i puntini di rumore"""
    S = max(2, round(args.residual_span * v.fps))
    pre_i = range(max(0, a - S), min(v.T, a + S // 2))
    post_i = range(max(0, b - S // 2), min(v.T, b + S))
    if len(pre_i) < 2 or len(post_i) < 2:
        return {"count": 0, "area_frac": 0.0}
    crop = lambda i: np.asarray(v.full(i).crop(box))
    pre = np.stack([crop(i) for i in pre_i])
    post = np.stack([crop(i) for i in post_i])
    bg_i = np.linspace(0, v.T - 1, min(30, v.T)).round().astype(int)
    bg = np.median(np.stack([crop(i) for i in bg_i]), axis=0).astype(np.uint8)
    bw, bh = box[2] - box[0], box[3] - box[1]
    sx1, sy1, sx2, sy2 = v.to_small(box)
    t = cv2.resize(thr[sy1:sy2, sx1:sx2].astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR)
    mpre = np.median(pre, axis=0).astype(np.uint8)
    mpost = np.median(post, axis=0).astype(np.uint8)
    stable = lambda st, m: (np.stack([cv2.absdiff(f, m).max(axis=2) for f in st]) < t).mean(axis=0) >= 0.7
    change = (cv2.absdiff(mpost, mpre).max(axis=2) > t) & stable(pre, mpre) & stable(post, mpost)
    change &= cv2.absdiff(mpost, bg).max(axis=2) > t
    subj = subject_mask(fg, post_i, args)[sy1:sy2, sx1:sx2]
    subj = cv2.resize(subj, (bw, bh), interpolation=cv2.INTER_NEAREST)
    # pixel di un oggetto piccolo possono essere sottili e sparsi: prima si uniscono i vicini, poi si
    # contano solo i gruppi abbastanza grandi (i puntini isolati di rumore restano piccoli e vengono scartati)
    change = (change & (subj == 0)).astype(np.uint8)
    grouped = cv2.dilate(change, np.ones((5, 5), np.uint8))
    n, lab, _, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    stats = np.zeros((n, 5), int)
    stats[:, 4] = np.bincount(lab[change > 0], minlength=n)  # area reale (pixel cambiati) di ogni gruppo
    full_area = v.W * v.H
    lo = max(4, args.residual_min_frac * full_area)
    small_objs = [s[4] for s in stats[1:] if lo <= s[4] <= args.residual_max_frac * full_area]
    return {"count": len(small_objs), "area_frac": round(float(sum(small_objs)) / full_area, 6)}


def priority(e: dict) -> float:
    """Punteggio solo per ORDINARE: sosta (max 1) + traccia lasciata (1.5). Nessun evento viene scartato."""
    stop = min(e["stop_s"] / 2.0, 1.0)
    trace = 1.0 if e["left_behind"]["count"] > 0 else 0.0
    return round(stop + 1.5 * trace, 3)


# ================================================================ disegno immagini

def render_grid(v: Video, idxs: list[int], box, args) -> Image.Image:
    cols = int(np.ceil(np.sqrt(len(idxs))))
    rows = int(np.ceil(len(idxs) / cols))
    gap = 4
    tw = (args.width - (cols - 1) * gap) // cols
    th = round(tw * (box[3] - box[1]) / (box[2] - box[0]))
    img = Image.new("RGB", (cols * tw + (cols - 1) * gap, rows * th + (rows - 1) * gap), (255, 255, 255))
    for k, i in enumerate(idxs):
        tile = v.full(i).crop(box).resize((tw, th), Image.LANCZOS)
        img.paste(tile, ((k % cols) * (tw + gap), (k // cols) * (th + gap)))
    return img


def render_overview(v: Video, tracks: list[dict]) -> Image.Image:
    img = v.full(v.T // 2).copy()
    d = ImageDraw.Draw(img)
    cmap = colormaps["tab10"]
    lw = max(2, v.W // 400)
    for tr in tracks:
        col = tuple(int(x * 255) for x in cmap(tr["id"] % 10)[:3])
        pts = [tuple(c) for c in tr["cents"]]
        if len(pts) > 1:
            d.line(pts, fill=col, width=lw)
        x, y, w, h = tr["boxes"][0]
        d.rectangle([x, y, x + w, y + h], outline=col, width=lw)
        label(img, f"#{tr['id']} {tr['start_s']}-{tr['end_s']}s", max(14, v.W // 70), pos=(x, max(0, y - 30)), color=col)
    return img


# ================================================================ main

def tiles_info(v: Video, idxs):
    return [{"n": k + 1, "frame": v.files[i].name, "t_s": round(i / v.fps, 2)} for k, i in enumerate(idxs)]


def process(folder: Path, args) -> None:
    fps = video_fps(folder, args.fps)
    v = Video(folder, fps, args.analysis_width)
    if v.T < 2:
        print(f"{folder.name}: troppo pochi frame, salto")
        return
    out = args.output / STRATEGY / folder.name
    out.mkdir(parents=True, exist_ok=True)

    cam = camera_check(v, args)
    (out / "camera.json").write_text(json.dumps(cam, indent=2), encoding="utf-8")
    print(f"{folder.name}: camera {'FISSA' if cam['fixed'] else 'IN MOVIMENTO'}  "
          f"(statico={cam['static_ratio']}, spostamento={cam['shift_median']})")

    if not cam["fixed"]:
        # ---- camera in movimento: frame intero, finestre su tutto il video
        win = max(2, round(args.window * fps))
        step = max(1, round(args.stride * fps))
        items = []
        (out / "grid_full").mkdir(exist_ok=True)
        for j, (a, b) in enumerate(windows(0, v.T - 1, v.T, win, step)):
            idxs = sample(a, b, args.tiles)
            name = f"win{j:02d}_{a / fps:05.1f}-{b / fps:05.1f}s.png"
            render_grid(v, idxs, v.content, args).save(out / "grid_full" / name)
            items.append({"image": name, "start_s": a / fps, "end_s": b / fps, "tiles": tiles_info(v, idxs)})
        (out / "grid_full" / "index.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
        print(f"   {len(items)} immagini grid_full")
        return

    # ---- camera fissa: soggetti -> eventi -> immagini
    fg, fg_raw, thr, bg = foreground(v, args)
    tracks = track(v, fg, args)
    (out / "tracks.json").write_text(json.dumps(
        [{k: t[k] for k in ("id", "start_s", "end_s", "max_size_frac", "stopped_s")} for t in tracks],
        indent=2), encoding="utf-8")
    render_overview(v, tracks).save(out / "overview.png")

    events = build_events(tracks, v, args)
    (out / "grid_crop").mkdir(exist_ok=True)
    items = []
    for k, e in enumerate(events):
        a, b, box = e["a"], e["b"], e["box"]
        e["left_behind"] = left_behind(v, fg_raw, thr, a, b, box, args)
        idxs = sample(a, b, args.tiles)
        name = f"ev{k:03d}_{a / fps:05.1f}-{b / fps:05.1f}s.png"
        render_grid(v, idxs, box, args).save(out / "grid_crop" / name)
        items.append({"image": name, "tracks": e["tracks"], "start_s": round(a / fps, 2), "end_s": round(b / fps, 2),
                      "crop": list(box), "stop_s": round(e["stop_s"], 2), "left_behind": e["left_behind"],
                      "priority": priority(e), "tiles": tiles_info(v, idxs)})
    (out / "grid_crop" / "index.json").write_text(json.dumps(items, indent=2), encoding="utf-8")

    ranked = sorted(items, key=lambda it: -it["priority"])
    (out / "priority.json").write_text(json.dumps(
        [{k: it[k] for k in ("image", "priority", "stop_s", "left_behind", "start_s", "end_s")} for it in ranked],
        indent=2), encoding="utf-8")
    print(f"   {len(tracks)} soggetti -> {len(events)} eventi. Priorità più alte:")
    for it in ranked[:3]:
        print(f"     {it['priority']:.2f}  {it['image']}  (sosta {it['stop_s']}s, traccia {it['left_behind']['count']})")


def preprocess_video(frames_dir, strips_dir, debug_root=None, fps=4.0, jpeg_quality=95, **overrides):
    """Applica la strategia ai frame di un video (frames_dir) e scrive le immagini per il VLM in strips_dir,
    con i nomi <video>_e<k>_t<primo>_t<ultimo>.jpg: gli indici sono quelli dei frame estratti, come per le
    strisce di compact_frames, cosi' main.list_strips le legge senza modifiche.
    Le immagini sono numerate in ordine di tempo (istante dell'ultimo riquadro), non di priorita': la
    priorita' resta in priority.json. Si scrive in strips_dir.tmp e si rinomina alla fine: se strips_dir
    esiste, e' completa (puo' anche essere vuota: camera fissa senza soggetti in movimento).
    debug_root: dove tenere camera.json, tracks.json, priority.json, overview.png e i png originali
    (None = non conservarli). overrides: le opzioni della riga di comando, ad es. width=1280."""
    frames_dir, strips_dir = Path(frames_dir), Path(strips_dir)
    video = frames_dir.name
    args = build_parser().parse_args([str(frames_dir)])
    args.fps = fps
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise TypeError(f"opzione sconosciuta: {key}")
        setattr(args, key, value)
    with tempfile.TemporaryDirectory() as scratch:
        args.output = Path(debug_root) if debug_root is not None else Path(scratch)
        out = args.output / STRATEGY / video
        shutil.rmtree(out, ignore_errors=True)  # niente file di esecuzioni precedenti
        process(frames_dir, args)
        subdir = next((d for d in ("grid_crop", "grid_full") if (out / d / "index.json").exists()), None)
        if subdir is None:
            raise ValueError(f"{video}: servono almeno 2 frame in {frames_dir}")
        items = json.loads((out / subdir / "index.json").read_text(encoding="utf-8"))
        frame_no = lambda tile: int(re.search(r"(\d+)$", Path(tile["frame"]).stem).group(1))
        events = sorted(((frame_no(it["tiles"][0]), frame_no(it["tiles"][-1]), it["image"]) for it in items),
                        key=lambda e: (e[1], e[0], e[2]))
        tmp = strips_dir.with_name(strips_dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        names = []
        for k, (first, last, image) in enumerate(events):
            name = f"{video}_e{k:03d}_t{first}_t{last}.jpg"
            with Image.open(out / subdir / image) as img:
                img.convert("RGB").save(tmp / name, quality=jpeg_quality)
            names.append(name)
        if strips_dir.exists():
            shutil.rmtree(strips_dir)
        tmp.rename(strips_dir)
    return [strips_dir / name for name in names]


def build_parser():
    p = argparse.ArgumentParser(description="Strategia 06: camera fissa/mobile -> eventi con priorità -> griglie per VLM")
    p.add_argument("input", type=Path, help="cartella frame di un video, oppure cartella che contiene più video")
    p.add_argument("--fps", type=float, default=4, help="fps dei frame se manca meta.json (default 4)")
    # fase 1
    p.add_argument("--static-min", type=float, default=0.6, help="frazione minima di pixel invariati per 'fissa'")
    p.add_argument("--shift-max", type=float, default=0.005, help="spostamento mediano max inquadratura (frazione larghezza)")
    # soggetti
    p.add_argument("--analysis-width", type=int, default=960, help="larghezza per l'analisi (default 960)")
    p.add_argument("--threshold", type=float, default=30, help="soglia differenza dallo sfondo (0-255)")
    p.add_argument("--noise-k", type=float, default=4, help="soglia più alta nelle zone rumorose")
    p.add_argument("--close-frac", type=float, default=0.015, help="quanto unire i pezzi di un soggetto (frazione larghezza)")
    p.add_argument("--min-area-frac", type=float, default=0.001, help="area minima soggetto (frazione del frame)")
    p.add_argument("--min-track", type=float, default=1.0, help="durata minima soggetto in secondi")
    p.add_argument("--max-gap", type=float, default=1.0, help="secondi di assenza tollerati nel tracciamento")
    p.add_argument("--max-jump", type=float, default=2.0, help="spostamento max tra frame, in dimensioni del soggetto")
    p.add_argument("--max-size-ratio", type=float, default=3.0, help="variazione max di area tra frame per lo stesso soggetto")
    p.add_argument("--color-w", type=float, default=1.0, help="peso del colore nel tracciamento")
    p.add_argument("--stop-speed", type=float, default=0.02, help="velocità sotto cui il soggetto è 'fermo' (diag/s)")
    # eventi e priorità
    p.add_argument("--merge-iou", type=float, default=0.3, help="sovrapposizione minima delle zone per unire eventi")
    p.add_argument("--merge-max-area", type=float, default=0.6, help="area max zona unita (frazione del frame)")
    p.add_argument("--residual-span", type=float, default=1.5, help="secondi usati prima/dopo per la 'traccia lasciata'")
    p.add_argument("--residual-min-frac", type=float, default=1e-5, help="area minima traccia (frazione frame, ~20px a 1080p)")
    p.add_argument("--residual-max-frac", type=float, default=0.01, help="area max traccia (frazione frame)")
    # immagini
    p.add_argument("--window", type=float, default=4, help="durata finestra in secondi (default 4)")
    p.add_argument("--stride", type=float, default=2, help="passo tra finestre in secondi (default 2)")
    p.add_argument("--tiles", type=int, default=9, help="riquadri per immagine (default 9)")
    p.add_argument("--width", type=int, default=1536, help="larghezza immagine finale")
    p.add_argument("--crop-pad", type=float, default=0.5, help="margine attorno al soggetto (frazione)")
    p.add_argument("--crop-min", type=float, default=0.3, help="dimensione minima ritaglio (frazione del frame)")
    p.add_argument("-o", "--output", type=Path, default=Path("output"), help="cartella radice output")
    return p


def main():
    args = build_parser().parse_args()

    folders = [args.input] if list_frames(args.input) else sorted(d for d in args.input.iterdir() if d.is_dir())
    for f in folders:
        process(f, args)


if __name__ == "__main__":
    main()
