"""Visual-JEV come misuratore: logit grezzi delle opzioni per molte domande sulla stessa immagine.

Riusa ChoiceScorer di main.py (modello 8 bit + adapter) senza modificarlo. Rispetto a ChoiceScorer.score():
- restituisce i logit grezzi di " A", " B", ... invece del solo softmax;
- un unico percorso (visione + prefisso una volta, poi i suffissi delle domande a blocchi di `chunk`), anche con 1 domanda;
- il blocco limita il picco di memoria della KV-cache copiata N volte: su GPU da 8 GB usare chunk <= 11 (oltre, la memoria
  va in spill e diventa 20-60 volte piu' lento); su A100 da 64 GB si puo' usare chunk >= numero di domande.
"""
import time

import numpy as np

from .paths import ensure_project_imports


def build_scorer(max_pixels):
    ensure_project_imports()
    import torch
    from transformers.cache_utils import DynamicCache
    import main as jev_main                  # main.py del progetto (carica torch/transformers/peft, fissa i seed)
    from main import ChoiceScorer

    class RawScorer(ChoiceScorer):
        @torch.inference_mode()
        def score_raw(self, image, questions, context="", chunk=11):
            """questions: lista di (testo, [opzioni]) oppure (testo, [opzioni], 'claim').
            Ritorna (logits [N, K_MAX] float32 con NaN oltre le opzioni, info con i tempi)."""
            vdm = self.model
            items = [{"qtype": (t[2] if len(t) > 2 else "choice"), "instruction": t[0], "candidates": list(t[1])} for t in questions]
            sync = torch.cuda.synchronize
            sync(); t0 = time.perf_counter()
            g = vdm.prepare_group(image, items, shared_context=context)
            sync(); t1 = time.perf_counter()
            base = vdm._prefill_prefix(g)                 # visione + prefisso: una volta per immagine
            sync(); t2 = time.perf_counter()
            n = g.n_questions
            out = None
            for lo in range(0, n, chunk):
                idx = list(range(lo, min(n, lo + chunk)))
                lg = self._suffix_batch(g, base, idx)
                if out is None:
                    out = np.full((n, lg.shape[1]), np.nan, dtype=np.float32)
                for r, i in enumerate(idx):
                    out[i, :g.n_options[i]] = lg[r, :g.n_options[i]]
            sync(); t3 = time.perf_counter()
            info = {"image_tokens": int((g.prefix_ids == g.image_token_id).sum()), "t_prepare": t1 - t0,
                    "t_prefill": t2 - t1, "t_suffix": t3 - t2, "n_q": n, "chunk": chunk}
            del base
            return out, info

        @staticmethod
        def _fork_once(cache, batch):
            """Come VDM._fork ma con una sola copia della KV del prefisso (.expand().contiguous())."""
            new = DynamicCache()
            for layer_idx, layer in enumerate(cache.layers):
                new.update(layer.keys.expand(batch, -1, -1, -1).contiguous(),
                           layer.values.expand(batch, -1, -1, -1).contiguous(), layer_idx)
            return new

        def _suffix_batch(self, g, base, idx):
            """Come VDM.run_prefix_share_batch, ma solo sulle domande `idx`."""
            vdm = self.model
            n, P = len(idx), g.prefix_len
            lens = [int(g.suffix_ids[i].shape[0]) for i in idx]
            S = max(lens)
            pad_id = vdm.tokenizer.pad_token_id or vdm.tokenizer.eos_token_id
            ids = torch.full((n, S), pad_id, dtype=torch.long)
            pos = torch.zeros(3, n, S, dtype=torch.long)
            attn = torch.zeros(n, P + S, dtype=torch.long)
            attn[:, :P] = 1
            for r, i in enumerate(idx):
                L = lens[r]
                ids[r, S - L:] = g.suffix_ids[i]
                pos[:, r, S - L:] = g.position_ids[i][:, P:]
                attn[r, P + S - L:] = 1
            cache = self._fork_once(base, n)
            emb, _ = vdm._embed(ids.to(vdm.device_str), None)
            res = vdm.vl_model.language_model(
                inputs_embeds=emb, position_ids=pos.to(vdm.device_str), attention_mask=attn.to(vdm.device_str),
                past_key_values=cache, use_cache=True)
            h = res.last_hidden_state[:, -1]
            lg = vdm._readout(h, [g.qtypes[i] for i in idx], [g.n_options[i] for i in idx])["lm_option_logits"]
            del cache, res
            return lg.float().cpu().numpy()

    scorer = RawScorer(max_pixels=max_pixels)
    scorer.jev_main = jev_main
    return scorer
