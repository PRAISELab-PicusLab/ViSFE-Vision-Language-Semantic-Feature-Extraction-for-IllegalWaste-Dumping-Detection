import csv
import glob
import json
import logging
import os
import re
import sys
import time

# Determinismo: da impostare prima di inizializzare CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "visual-jev", "code"))
from vdm.models.vdm_model import VDM

from decompose_video import FPS_OUT
from progress_utils import timed

SEED = 0
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

SCRIPT_DIR = os.path.join(ROOT, "script")
FRAMES_DIR = "compacted_frames/vid0004"  # sovrascritto da: python main.py <cartella>
RESULTS_DIR = os.path.join(ROOT, "results")
LABELS_GLOB = os.path.join(ROOT, "dataset", "*", "labels", "{video}.json")
MAX_PIXELS = 1003520

BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
ADAPTER = "guanxuyu/visual-jev-4b-answer-sft"

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)


def load_vdm_8bit(device="cuda"):
    # VDM.__init__ carica il backbone in bf16 con .to(device), incompatibile con i pesi quantizzati:
    # costruiamo lo stesso oggetto caricando il backbone quantizzato (vision tower e lm_head restano in bf16).
    # QUANT_BITS=8 (default, l'ambiente validato) oppure QUANT_BITS=4: su alcune macchine locali con poca RAM
    # il caricamento 8-bit mappa in memoria l'intero file dei pesi e puo' esaurire la memoria virtuale di Windows
    # (errore di sistema 1455, "il file di paging e' troppo piccolo"); il percorso 4-bit (NF4) carica con un
    # picco di memoria CPU/VRAM inferiore ed e' quindi un fallback utile quando capita quell'errore. Le probabilita'
    # cambiano leggermente rispetto all'8-bit; di default resta 8-bit per non alterare l'ambiente gia' validato.
    quant_bits = os.environ.get("QUANT_BITS", "8")
    if quant_bits == "4":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    llm_int8_skip_modules=["visual", "lm_head"])
    else:
        quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual", "lm_head"])
    vdm = VDM.__new__(VDM)
    nn.Module.__init__(vdm)
    vdm.model_path = BASE_MODEL
    vdm.device_str = device
    vdm.processor = AutoProcessor.from_pretrained(BASE_MODEL)
    vdm.tokenizer = vdm.processor.tokenizer
    vdm.config = AutoConfig.from_pretrained(BASE_MODEL)
    vdm.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=quant,
        device_map={"": 0},
    ).eval()
    vdm.heads = None
    vdm.image_token_id = vdm.config.image_token_id
    vdm.keep_full_lm_logits = False
    vdm._option_ids = vdm._build_option_ids()
    return vdm


def _conv3d_fp32_on_pre_ampere(model):
    """Su GPU senza bf16 nativo (es. T4, capability < 8) la Conv3d del vision tower in bf16 e' lentissima
    (decine di secondi a striscia). La eseguiamo in fp32 e riconvertiamo in bf16. Le probabilita' cambiano di poco
    (misurato su vid0001: fino a ~0.025 rispetto al percorso bf16 di T4, risposte invariate). Sulle GPU con bf16
    nativo non cambia nulla. Per disattivarla: variabile d'ambiente WV_CONV3D_FP32=0."""
    if os.environ.get("WV_CONV3D_FP32", "1") == "0":
        return 0
    if torch.cuda.get_device_capability()[0] >= 8:
        return 0
    n = 0
    for module in model.modules():
        if isinstance(module, nn.Conv3d) and module.weight.dtype == torch.bfloat16:
            def forward(x, module=module):
                bias = None if module.bias is None else module.bias.float()
                return module._conv_forward(x.float(), module.weight.float(), bias).to(torch.bfloat16)
            module.forward = forward
            n += 1
    print(f"Conv3d eseguite in fp32 (GPU senza bf16 nativo): {n} moduli", flush=True)
    return n

class ChoiceScorer:
    def __init__(self, device="cuda", max_pixels=200704):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU NVIDIA (CUDA) non disponibile: il modello non viene eseguito su CPU")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        quant_bits = os.environ.get("QUANT_BITS", "8")
        with timed(f"Caricamento Qwen3-VL {quant_bits}-bit"):
            self.model = load_vdm_8bit(device)
        # Qwen3-VL espone `size` come dizionario, ad es.
        # {"shortest_edge": 65536, "longest_edge": 16777216}.
        # `longest_edge` rappresenta il budget massimo di pixel per immagine.
        size = self.model.processor.image_processor.size
        if isinstance(size, dict):
            size["longest_edge"] = max_pixels
        elif hasattr(size, "longest_edge"):
            # Compatibilita' con eventuali versioni in cui `size` e' un oggetto.
            size.longest_edge = max_pixels
        else:
            raise TypeError(
                f"Formato inatteso per image_processor.size: {type(size).__name__} ({size!r})"
            )
        with timed("Caricamento adapter LoRA"):
            self.model.backbone = PeftModel.from_pretrained(self.model.backbone, ADAPTER).eval()
        _conv3d_fp32_on_pre_ampere(self.model.backbone)

    @torch.inference_mode()
    def score(self, image, questions, context=""):
        """questions: list of (question, [choices]). Returns a list of {choice: probability}."""
        if isinstance(image, str):
            with Image.open(image) as src:
                image = src.convert("RGB")
        items = [{"qtype": "choice", "instruction": q, "candidates": list(c)} for q, c in questions]
        group = self.model.prepare_group(image, items, shared_context=context)
        run = self.model.run_independent if len(items) == 1 else self.model.run_prefix_share_batch
        out = run(group)
        results = []
        for i, (_, choices) in enumerate(questions):
            logits = out["lm_option_logits"][i, : group.n_options[i]]
            probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            results.append(dict(zip(choices, probs)))
        return results


def crop_margins(im):
    # I margini bianchi esterni della striscia consumano pixel del budget MAX_PIXELS; gli spazi tra i frame restano.
    filled = np.asarray(im.convert("L")) < 250
    rows = np.where(filled.any(axis=1))[0]
    cols = np.where(filled.any(axis=0))[0]
    return im.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def list_strips(frames_dir):
    """Strisce {video}_t{inizio}_t{fine}.jpg in ordine temporale; t e' l'indice del frame estratto a FPS_OUT."""
    strips = []
    for path in glob.glob(os.path.join(frames_dir, "*.jpg")):
        m = re.search(r"_t(\d+)_t(\d+)\.jpg$", path)
        if m:
            strips.append((int(m.group(1)), int(m.group(2)), path))
    return sorted(strips)


def load_label(video):
    paths = glob.glob(LABELS_GLOB.format(video=video))
    if not paths:
        return None
    with open(paths[0], encoding="utf-8") as f:
        return json.load(f)


def read_txt(name):
    with open(os.path.join(SCRIPT_DIR, name), encoding="utf-8") as f:
        return f.read().strip()


def load_prompt():
    """(context, question, choices) dai txt in script/."""
    choices = [line.strip() for line in read_txt("choices.txt").splitlines() if line.strip()]
    return read_txt("context.txt"), read_txt("question.txt"), choices


def classify_strips(scorer, frames_dir, prompt):
    """Classifica ogni striscia della cartella in ordine temporale; restituisce una riga per striscia alla volta."""
    context, question, choices = prompt
    bar = tqdm(list_strips(frames_dir), desc=f"Inferenza {os.path.basename(os.path.normpath(frames_dir))}", unit="striscia")
    for t_start, t_end, path in bar:
        t0 = time.perf_counter()
        with Image.open(path) as src:
            image = crop_margins(src.convert("RGB"))
        t1 = time.perf_counter()
        (probs,) = scorer.score(image, [(question, choices)], context=context)
        bar.set_postfix_str(f"immagine {t1 - t0:.2f}s, modello {time.perf_counter() - t1:.2f}s")
        yield {"image": os.path.basename(path), "t_start_s": t_start / FPS_OUT, "t_end_s": t_end / FPS_OUT,
               "answer": max(probs, key=probs.get), **{f"p_{c}": round(p, 4) for c, p in probs.items()}}


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image", "t_start_s", "t_end_s", "answer"])
        writer.writeheader()
        writer.writerows(rows)


def first_detection(rows, key="answer"):
    # Come nella challenge conta solo il primo rilevamento, emesso quando l'ultimo frame della striscia e' disponibile.
    # key="answer" -> Visual-JEV; key="gemma_answer" -> ablation (solo l'assessment di Gemma, senza JEV).
    return min((r["t_end_s"] for r in rows if r.get(key) == "yes"), default=None)


def evaluate(label, detection):
    """Esito della challenge per un video: TP, FP, FN o TN (finestra valida [g-3 s, g+10 s])."""
    if label["label"] != "true":
        return "FP" if detection is not None else "TN"
    if detection is None:
        return "FN"
    g = float(label["timestamp"])
    return "TP" if g - 3 <= detection <= g + 10 else "FP"


def main():
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else FRAMES_DIR
    video = os.path.basename(os.path.normpath(frames_dir))
    if not list_strips(frames_dir):
        raise FileNotFoundError(f"Nessuna striscia *_t<inizio>_t<fine>.jpg in {frames_dir}")

    scorer = ChoiceScorer(max_pixels=MAX_PIXELS)
    rows = []
    for row in classify_strips(scorer, frames_dir, load_prompt()):
        rows.append(row)
        probs = "  ".join(f"{k[2:]}={v:.3f}" for k, v in row.items() if k.startswith("p_"))
        print(f"{row['image']:<24} {row['t_start_s']:6.2f}-{row['t_end_s']:6.2f}s  {row['answer']:<4} {probs}", flush=True)

    csv_path = os.path.join(RESULTS_DIR, f"{video}.csv")
    save_csv(rows, csv_path)
    detection = first_detection(rows)
    n_yes = sum(r["answer"] == "yes" for r in rows)
    print(f"\n{n_yes}/{len(rows)} strisce classificate yes. Risultati in {csv_path}")
    print("Primo rilevamento:", f"{detection:.2f} s" if detection is not None else "nessuno")

    label = load_label(video)
    if label is not None:
        where = f" a {float(label['timestamp']):g} s" if label["label"] == "true" else ""
        print(f"Etichetta: {label['metadata']['event_type']}{where} -> {evaluate(label, detection)}")


if __name__ == "__main__":
    main()
