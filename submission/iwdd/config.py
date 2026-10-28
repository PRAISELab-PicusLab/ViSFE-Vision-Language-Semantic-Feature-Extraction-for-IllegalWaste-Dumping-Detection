"""Caricamento della configurazione (config/default.json) con sovrascritture da riga di comando."""
import copy
import json
from pathlib import Path

from .paths import CONFIG_DIR


def load_config(path=None, overrides=None):
    cfg = json.loads(Path(path or CONFIG_DIR / "default.json").read_text(encoding="utf-8-sig"))
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def snapshot(cfg):
    return copy.deepcopy(cfg)
