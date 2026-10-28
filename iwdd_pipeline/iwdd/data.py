"""Dataset MIVIA-IWDD: elenco dei video, etichette, sottoinsiemi stratificati e suddivisione in shard."""
import json
import random
from dataclasses import dataclass

from .paths import DATASET

SPLIT_DIRS = {"train": "train", "test": "test"}
EVENT_TYPES = ("normal", "static dumping", "dynamic dumping")


@dataclass(frozen=True)
class Video:
    vid: str
    split: str
    is_pos: bool
    g: float            # secondo dello scarico (-1 nei negativi)
    event_type: str
    duration: float     # durata dichiarata nel JSON (s)
    time_of_day: str
    resolution: str

    @property
    def path(self):
        return DATASET / SPLIT_DIRS[self.split] / "videos" / f"{self.vid}.mp4"

    def label(self):
        """Formato di main.evaluate(): label come stringa 'true'/'false' e timestamp."""
        return {"label": "true" if self.is_pos else "false", "timestamp": self.g}


def load_split(split):
    ldir = DATASET / SPLIT_DIRS[split] / "labels"
    if not ldir.is_dir():
        raise FileNotFoundError(f"Cartella etichette non trovata: {ldir} (controlla IWDD_DATASET)")
    out = []
    for p in sorted(ldir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        m = d["metadata"]
        out.append(Video(vid=p.stem, split=split, is_pos=str(d["label"]).lower() == "true",
                         g=float(d["timestamp"]), event_type=m["event_type"], duration=float(m.get("duration", 0.0)),
                         time_of_day=m.get("time_of_day", ""), resolution=m.get("resolution", "")))
    return out


def stratified_limit(videos, n, seed=0):
    """n video con le proporzioni del dataset (50% normal, 25% static, 25% dynamic), scelti con seed fisso."""
    if n is None or n >= len(videos):
        return list(videos)
    rng = random.Random(seed)
    quota = {"normal": n // 2, "static dumping": n // 4}
    quota["dynamic dumping"] = n - sum(quota.values())
    chosen = []
    for t, k in quota.items():
        pool = sorted((v for v in videos if v.event_type == t), key=lambda v: v.vid)
        rng.shuffle(pool)
        chosen += pool[:k]
    return sorted(chosen, key=lambda v: v.vid)


def get_videos(splits, limits=None, seed=0):
    """Video dei split richiesti. limits: {'train': N, 'test': M} per run di prova."""
    out = []
    for s in splits:
        vs = load_split(s)
        lim = (limits or {}).get(s)
        out += stratified_limit(vs, lim, seed)
    return out


def shard(videos, index, num, weights=None):
    """Assegna i video a `num` shard bilanciando i pesi (es. numero di immagini): greedy LPT, deterministico.
    Ritorna i video dello shard `index`."""
    if num <= 1:
        return list(videos)
    w = weights or {}
    order = sorted(videos, key=lambda v: (-w.get(v.vid, 1), v.vid))
    loads = [0.0] * num
    bins = [[] for _ in range(num)]
    for v in order:
        i = min(range(num), key=lambda k: (loads[k], k))
        bins[i].append(v)
        loads[i] += w.get(v.vid, 1)
    return sorted(bins[index], key=lambda v: v.vid)
