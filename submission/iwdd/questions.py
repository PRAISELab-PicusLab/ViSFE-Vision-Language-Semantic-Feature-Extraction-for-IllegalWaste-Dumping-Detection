"""Set di domande bloccato (config/questions.json) e contesto dato a Visual-JEV."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .paths import CONFIG_DIR

KINDS = ("yesno", "claim", "multi")
YESNO = ["yes", "no"]


@dataclass(frozen=True)
class Question:
    id: str
    kind: str                      # yesno | claim | multi
    en: str                        # testo esatto passato al modello
    it: str                        # solo per leggibilita'
    group: str
    origin: str
    options: tuple                 # opzioni in inglese
    options_it: tuple = ()
    feature_options: tuple = ()    # per claim/multi: opzioni usate come feature

    @property
    def n_options(self):
        return len(self.options)

    @property
    def qtype(self):
        return "claim" if self.kind == "claim" else "choice"

    def jev_item(self):
        """(testo, opzioni[, 'claim']) come lo vuole RawScorer.score_raw."""
        return (self.en, list(self.options)) + (("claim",) if self.kind == "claim" else ())


def _require_english(text, where):
    """Garanzia: a Visual-JEV arriva SOLO testo inglese. Il file delle domande deve essere ASCII (niente lettere accentate)."""
    if not text.isascii():
        raise ValueError(f"Testo non inglese (caratteri non ASCII) in {where}: {text!r}. "
                         "Le domande per Visual-JEV devono essere in inglese; le traduzioni vanno in questions_it.json.")


def load_questions(path=None):
    """Domande da config/questions.json (SOLO inglese, e' cio' che riceve il modello). Le traduzioni italiane, se presenti nel file
    gemello <nome>_it.json, servono solo alla leggibilita' (QUESTIONS.md) e non vengono mai passate a Visual-JEV."""
    qpath = Path(path) if path else CONFIG_DIR / "questions.json"
    d = json.loads(qpath.read_text(encoding="utf-8-sig"))
    it_path = qpath.with_name(qpath.stem + "_it.json")
    it_map = json.loads(it_path.read_text(encoding="utf-8-sig")).get("questions", {}) if it_path.exists() else {}
    claim_opts = tuple(d["claim_options"])
    out = []
    for q in d["questions"]:
        kind = q["kind"]
        assert kind in KINDS, q
        tr = it_map.get(q["id"], {})
        if kind == "yesno":
            opts, opts_it, feat = tuple(YESNO), ("sì", "no"), ()
        elif kind == "claim":
            opts, opts_it, feat = claim_opts, ("supportata", "contraddetta", "non determinata"), tuple(q.get("feature_options", [0]))
        else:
            opts, opts_it, feat = tuple(q["options_en"]), tuple(tr.get("options_it", [])), tuple(q["feature_options"])
        _require_english(q["en"], f"{q['id']}.en")
        for o in opts:
            _require_english(o, f"{q['id']}.options")
        out.append(Question(id=q["id"], kind=kind, en=q["en"], it=tr.get("it", ""), group=q.get("group", ""),
                            origin=q.get("origin", ""), options=opts, options_it=opts_it, feature_options=feat))
    ids = [q.id for q in out]
    assert len(ids) == len(set(ids)), "id di domanda duplicati"
    return out


def load_derived(path=None):
    d = json.loads(Path(path or CONFIG_DIR / "questions.json").read_text(encoding="utf-8-sig"))
    return {k: v for k, v in d.get("derived_features", {}).items() if not k.startswith("_")}


def questions_hash(questions, context_mode, max_pixels):
    """Impronta di cio' che determina le feature: testo, opzioni, tipo, contesto, risoluzione. Va nel manifest delle feature."""
    blob = json.dumps([[q.id, q.kind, q.en, list(q.options)] for q in questions] + [context_mode, max_pixels], ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def context_text(mode, dt):
    """Testo fra immagine e domanda (identico per tutte le domande). mode: none | basic | rows. dt = secondi fra due frame."""
    if mode == "none":
        return ""
    base = ("The image shows 9 video frames, arranged in a 3x3 grid in chronological order: left to right, then top to bottom. ")
    if mode == "rows":
        base += "The top row shows the first three frames, the middle row the next three, and the bottom row the last three. "
    elif mode != "basic":
        raise ValueError(f"context_mode sconosciuto: {mode}")
    return (base + "The frames may be zoomed in on the area where people or objects move, so the view can be cropped. "
            f"Consecutive frames are about {dt:.1f} seconds apart. "
            "Answer only from what is visible in the frames. Do not assume events that are not observed.")


def _vdm_prompts():
    """Il modulo prompts.py di Visual-JEV caricato per percorso (e' testo puro, senza torch): e' lo stesso codice che costruisce
    il prompt vero in VDM.prepare_group(), quindi l'audit mostra esattamente cio' che vede il modello."""
    import importlib.util

    from .paths import ensure_project_imports
    vdm_dir = ensure_project_imports()
    if vdm_dir is None:
        raise FileNotFoundError("Codice di Visual-JEV (vdm) non trovato: controlla IWDD_PROJECT")
    spec = importlib.util.spec_from_file_location("vdm_prompts_audit", Path(vdm_dir) / "vdm" / "models" / "prompts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_prompts(questions, context_mode, dt=0.5):
    """[(id, prompt completo)]: system + immagine (segnaposto) + contesto + 'Question:' + opzioni, come lo riceve Visual-JEV.
    Per le affermazioni il codice di Visual-JEV usa le sue tre opzioni standard ('supported by the image', ...)."""
    pm = _vdm_prompts()
    ctx = context_text(context_mode, dt)
    out = []
    for q in questions:
        opts = pm.CLAIM_OPTIONS if q.kind == "claim" else list(q.options)
        out.append((q.id, pm.prefix_text(ctx) + pm.suffix_text(q.en, opts, q.qtype)))
    return out


def write_questions_md(questions, path):
    lines = [f"# Domande date a Visual-JEV ({len(questions)})", "",
             "Testo inglese = quello esatto passato al modello. Le opzioni le aggiunge il codice di Visual-JEV "
             "(`A. yes / B. no`, oppure le opzioni indicate). La traduzione italiana e' solo per leggibilita'.", "",
             "| # | id | tipo | gruppo | origine | inglese (al modello) | italiano |", "|---|---|---|---|---|---|---|"]
    kind_it = {"yesno": "sì/no", "claim": "affermazione", "multi": "scelta multipla"}
    for i, q in enumerate(questions, 1):
        extra = ""
        if q.kind == "claim":
            extra = "<br>(supported / contradicted / not determined)"
        elif q.kind == "multi":
            extra = "<br>" + "<br>".join(f"{chr(65 + k)}. {o} / {q.options_it[k] if k < len(q.options_it) else ''}"
                                         for k, o in enumerate(q.options))
        lines.append(f"| {i} | `{q.id}` | {kind_it[q.kind]} | {q.group} | {q.origin} | {q.en}{extra} | {q.it} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
