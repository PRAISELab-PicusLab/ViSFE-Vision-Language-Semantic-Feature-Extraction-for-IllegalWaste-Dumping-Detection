"""Interfaccia a riga di comando: python -m iwdd <comando> [opzioni]   (da iwdd_pipeline/)."""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import load_config
from .data import get_videos, shard
from .paths import CLASSIFIERS_DIR, DATASET, PIPELINE_DIR, PROJECT_ROOT, RUNS, Run, classifier_out, ensure_project_imports
from .questions import load_derived, load_questions, write_questions_md

log = logging.getLogger("iwdd")
EXIT_TIME_BUDGET = 75      # EX_TEMPFAIL: l'estrazione si e' fermata per il budget di tempo, c'e' ancora lavoro


def _common(p):
    p.add_argument("--run", default="main", help="nome del run: cartella <IWDD_RUNS>/<run> (default: main)")
    p.add_argument("--splits", default="train,test", help="split da usare (default: train,test)")
    p.add_argument("--limit-train", type=int, default=None, help="solo N video di train (stratificati, seed fisso): per prove")
    p.add_argument("--limit-test", type=int, default=None, help="solo N video di test (stratificati, seed fisso): per prove")
    p.add_argument("--config", default=None, help="config alternativa (default: config/default.json)")
    p.add_argument("--questions", default=None, help="file domande alternativo (default: config/questions.json)")
    p.add_argument("--part", type=int, default=0, help="solo la parte K dei video (0..N-1): un task di un job array")
    p.add_argument("--num-parts", type=int, default=1, help="in quante parti dividere i video (bilanciate sulla durata)")


def _default_jobs():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)


def _videos(a, splits=None, partition=True):
    """Video dei split richiesti; con --num-parts > 1 solo quelli della parte --part (stessa divisione per scan/prepare/extract:
    deterministica e bilanciata sulla durata dichiarata nelle etichette, quindi non dipende dai passi precedenti)."""
    sp = [s for s in (splits if splits is not None else a.splits).split(",") if s]
    vids = get_videos(sp, {"train": a.limit_train, "test": a.limit_test})
    if partition and a.num_parts > 1:
        if not 0 <= a.part < a.num_parts:
            raise SystemExit(f"--part {a.part} fuori da 0..{a.num_parts - 1}")
        vids = shard(vids, a.part, a.num_parts, {v.vid: max(v.duration, 1.0) for v in vids})
        log.info("parte %d di %d: %d video", a.part, a.num_parts, len(vids))
    return vids


def _reuse(run, arg):
    """Cartelle di immagini evento da riusare (sola lettura). Si salvano nel run per i comandi successivi."""
    f = run.root / "run_config.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    if arg is not None:
        d["reuse_events"] = [x for x in arg.split(",") if x]
        run.root.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(d, indent=1))
    return d.get("reuse_events", [])


# ---------------------------------------------------------------------------------------------------------------------- comandi
def cmd_doctor(a):
    import importlib
    import platform
    ok = True
    print(f"python {platform.python_version()} ({sys.executable})")
    print(f"progetto : {PROJECT_ROOT}\ndataset  : {DATASET}\nrisultati: {RUNS}")
    for f in ("main.py", "preprocessing.py", "preprocessing_pt2.py", "decompose_video.py", "progress_utils.py"):
        e = (PROJECT_ROOT / f).is_file()
        ok &= e
        print(f"  [{'ok' if e else 'MANCA'}] {f}")
    vdm = ensure_project_imports()
    ok &= vdm is not None
    print(f"  [{'ok' if vdm else 'MANCA'}] codice Visual-JEV (vdm): {vdm}")
    for s in ("train", "test"):
        n = len(list((DATASET / s / "videos").glob("*.mp4"))) if (DATASET / s / "videos").is_dir() else 0
        nl = len(list((DATASET / s / "labels").glob("*.json"))) if (DATASET / s / "labels").is_dir() else 0
        ok &= n > 0 and n == nl
        print(f"  [{'ok' if n and n == nl else 'MANCA'}] dataset/{s}: {n} video, {nl} etichette")
    for m in ("numpy", "pandas", "pyarrow", "sklearn", "scipy", "matplotlib", "cv2", "PIL", "joblib", "torch", "transformers", "peft", "bitsandbytes"):
        try:
            mod = importlib.import_module(m)
            print(f"  [ok] {m} {getattr(mod, '__version__', '')}")
        except Exception as e:
            ok &= m not in ("torch", "transformers", "peft", "bitsandbytes", "cv2", "numpy", "pandas", "pyarrow", "sklearn")
            print(f"  [{'MANCA' if m in ('torch', 'transformers', 'peft', 'bitsandbytes', 'cv2') else 'manca'}] {m}: {type(e).__name__}")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.0f} GiB)")
        else:
            print("  GPU: NON disponibile (serve per 'extract')")
    except Exception:
        pass
    try:
        from huggingface_hub import snapshot_download
        for repo in ("Qwen/Qwen3-VL-4B-Instruct", "guanxuyu/visual-jev-4b-answer-sft"):
            try:
                snapshot_download(repo, local_files_only=True)
                print(f"  [ok] modello in cache: {repo}")
            except Exception:
                ok = False
                print(f"  [MANCA] modello NON in cache: {repo}  (si scarica dall'Hugging Face Hub al primo avvio)")
    except Exception as e:
        print(f"  huggingface_hub non disponibile: {e}")
    qs = load_questions(a.questions)
    print(f"  domande: {len(qs)} ({sum(q.kind == 'yesno' for q in qs)} sì/no, {sum(q.kind == 'claim' for q in qs)} affermazioni, "
          f"{sum(q.kind == 'multi' for q in qs)} scelta multipla)")
    print("TUTTO OK" if ok else "CI SONO PROBLEMI (righe MANCA)")
    return 0 if ok else 1


def cmd_questions(a):
    qs = load_questions(a.questions)
    out = Path(a.out) if a.out else PIPELINE_DIR / "QUESTIONS.md"
    write_questions_md(qs, out)
    print(f"{len(qs)} domande scritte in {out}")
    return 0


def cmd_prompts(a):
    """Stampa il prompt ESATTO che Visual-JEV riceve per ogni domanda (con il contesto scelto)."""
    from .questions import render_prompts
    cfg = load_config(a.config)
    qs = load_questions(a.questions)
    mode = a.context_mode or cfg["jev"]["context_mode"]
    ids = set(a.only.split(",")) if a.only else None
    for qid, txt in render_prompts(qs, mode):
        if ids is None or qid in ids:
            print(f"##### {qid}\n{txt}\n")
    return 0


def cmd_scan(a):
    from .scan import save_groups, scan_all
    cfg = load_config(a.config)
    run = Run(a.run)
    vids = _videos(a)
    errors = scan_all(run, vids, cfg, a.workers)
    if a.num_parts > 1:
        log.info("scansione di una parte: i gruppi di camera si calcolano dopo su tutti i video (comando 'groups')")
    else:
        save_groups(run, vids, cfg)
    return 1 if errors else 0


def cmd_groups(a):
    """Gruppi di camera su tutti i video scansionati (serve dopo una scansione divisa in parti)."""
    from .scan import save_groups
    save_groups(Run(a.run), _videos(a, partition=False), load_config(a.config))
    return 0


def cmd_prepare(a):
    from .events import prepare_all
    cfg = load_config(a.config)
    run = Run(a.run)
    reuse = _reuse(run, a.reuse_events)
    errors = prepare_all(run, _videos(a), cfg, a.workers, reuse)
    return 1 if errors else 0


def cmd_extract(a):
    from .extract import extract
    cfg = load_config(a.config)
    run = Run(a.run)
    reuse = _reuse(run, a.reuse_events)
    qs = load_questions(a.questions)
    # con --num-parts la divisione e' gia' fatta da _videos: dentro extract un solo shard, con l'indice della parte nei nomi dei file
    shard_idx, num_shards = (a.part, 1) if a.num_parts > 1 else (a.shard, a.num_shards)
    res = extract(run, _videos(a), qs, cfg, shard_idx, num_shards, a.chunk, a.max_pixels, a.context_mode, a.max_minutes, a.force,
                  reuse, a.limit_videos, getattr(a, "wait_for", None))
    log.info("estrazione: %s", res)
    if res["remaining"]:
        return EXIT_TIME_BUDGET
    return 1 if res["errors"] or res["missing"] else 0


def cmd_status(a):
    """Quanti video sono stati scansionati, preparati ed estratti, per split."""
    from .events import images_dir, list_event_images
    run = Run(a.run)
    reuse = _reuse(run, None)
    print(f"run {run.name}: {run.root}")
    ok = True
    for s in [x for x in a.splits.split(",") if x]:
        vids = get_videos([s], {"train": a.limit_train, "test": a.limit_test})
        scanned = sum((run.root / "meta" / f"{v.vid}.json").exists() for v in vids)
        dirs = {v.vid: images_dir(run, v.vid, reuse) for v in vids}
        prepared = sum(d is not None for d in dirs.values())
        n_img = sum(len(list_event_images(d)) for d in dirs.values() if d)
        extracted = sum((run.root / "features" / f"{v.vid}.parquet").exists() for v in vids)
        no_ev = sum(1 for d in dirs.values() if d is not None and not list_event_images(d))
        print(f"  {s:<5}: {len(vids)} video | scansionati {scanned} | immagini evento pronte {prepared} ({n_img} griglie, {no_ev} video senza griglie) "
              f"| feature estratte {extracted}")
        ok &= extracted == len(vids)
    print("COMPLETO" if ok else "INCOMPLETO: mancano feature (esegui 'prepare' e 'extract')")
    return 0


def cmd_train_eval(a):
    from .models import load_spec
    from .report import write_report
    from .scan import save_groups
    from .train_eval import run_train_eval
    cfg = load_config(a.config)
    spec = load_spec(a.classifier) if a.classifier else None
    run = Run(a.run, classifier=a.classifier)
    qs, derived = load_questions(a.questions), load_derived(a.questions)
    tr = _videos(a, a.train_splits, partition=False)
    te = _videos(a, a.test_splits, partition=False)
    if not (run.meta / "groups.json").exists():
        save_groups(run, tr + te, cfg)
    run_train_eval(run, cfg, qs, derived, tr, te, a.time_col, a.model_tag, a.allow_partial, spec, a.jobs or _default_jobs())
    print("report:", write_report(run, qs))
    return 0


def cmd_report(a):
    from .report import write_report
    print(write_report(Run(a.run, classifier=a.classifier), load_questions(a.questions)))
    return 0


def cmd_classifiers(a):
    from .models import list_classifiers, load_spec
    for name in list_classifiers():
        s = load_spec(name)
        print(f"{name:<16} kind={s['kind']:<14} griglia video {len(s['video_grid']):>2} | finestre {len(s['window_grid']):>2} | "
              f"{s.get('description', '')}")
    return 0


def _summary_row(name, run_name):
    """Una riga di confronto dal summary.json di un classificatore (o di una coppia A+B); None se non c'e'."""
    sp = classifier_out(name, run_name) / "metrics" / "summary.json"
    if not sp.exists():
        return None
    s = json.loads(sp.read_text(encoding="utf-8"))
    cv, te, sel = s["cv_train"], s.get("test") or {}, s["selection"]
    ci = (te.get("ci95") or {}).get("f1_challenge", [float("nan")] * 2)
    return {"classificatore": name, "kind": s.get("kind"),
            "F1 CV": cv["f1_challenge"], "AUC CV": cv["auc_detect"],
            "F1 test": te.get("f1_challenge"), "F1 test IC95 basso": ci[0], "F1 test IC95 alto": ci[1],
            "P test": te.get("precision"), "R test": te.get("recall"), "TP": te.get("TP"), "FP": te.get("FP"),
            "FN": te.get("FN"), "TN": te.get("TN"), "AUC test": te.get("auc_detect"), "F1 rilevamento test": te.get("f1_detect"),
            "feature": sel["feature_set"], "aggregazioni": "+".join(sel["aggs"]),
            "parametri video": json.dumps(sel.get("video_params"), sort_keys=True),
            "parametri finestre": json.dumps(sel.get("window_params"), sort_keys=True),
            "delta": sel["delta"], "soglia": sel["threshold"]}


def _fmt(x):
    return "–" if x is None or x != x else f"{x:.3f}"


def _table_md(df):
    L = ["| classificatore | F1 CV | AUC CV | **F1 test** [IC95] | P / R test | TP / FP / FN / TN | AUC test | parametri video | parametri finestre | delta | soglia |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = _fmt
    for r in df.to_dict("records"):
        L.append(f"| `{r['classificatore']}` | {f(r['F1 CV'])} | {f(r['AUC CV'])} | **{f(r['F1 test'])}** "
                 f"[{f(r['F1 test IC95 basso'])}, {f(r['F1 test IC95 alto'])}] | {f(r['P test'])} / {f(r['R test'])} | "
                 f"{r['TP']} / {r['FP']} / {r['FN']} / {r['TN']} | {f(r['AUC test'])} | `{r['parametri video']}` | "
                 f"`{r['parametri finestre']}` | {r['delta']:+d} | {r['soglia']:.3f} |")
    return L


def cmd_compare(a):
    """Tabella di confronto dei classificatori addestrati sullo stesso run: classifiers/comparison/<run>.{csv,md}.
    Con --combinations: le coppie A+B (si'/no di A, tempo di B) -> classifiers/comparison/<run>_combinations.{csv,md}."""
    import pandas as pd
    from .models import list_classifiers
    clfs = list_classifiers()
    names = [f"{v}+{w}" for v in clfs for w in clfs] if a.combinations else clfs
    rows = []
    for name in names:
        r = _summary_row(name, a.run)
        if r is None:
            log.warning("%s: nessun risultato per il run %s", name, a.run)
        else:
            rows.append(r)
    if not rows:
        print(f"Nessun risultato per il run {a.run}")
        return 1
    df = pd.DataFrame(rows).sort_values("F1 CV", ascending=False)
    out = Path(os.environ.get("IWDD_CLASSIFIER_OUT") or CLASSIFIERS_DIR) / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.run}_combinations" if a.combinations else a.run
    note = ("Ordinati per F1 della challenge in CV (out-of-fold sul train): e' il criterio legittimo per scegliere. Il test misura e "
            "basta: scegliere il modello guardando il test lo renderebbe ottimistico.")
    if not a.combinations:
        L = [f"# Confronto dei classificatori — run `{a.run}`", "", note, ""] + _table_md(df)
        L += ["", "Dettagli di ogni modello: `classifiers/<nome>/runs/" + a.run + "/report.md`.", ""]
        cols = ["classificatore", "F1 CV", "AUC CV", "F1 test", "AUC test"]
    else:
        df["si/no"] = df["classificatore"].str.split("+").str[0]
        df["tempo"] = df["classificatore"].str.split("+").str[1]
        order = [c for c in clfs if c in set(df["si/no"]) | set(df["tempo"])]
        L = [f"# Combinazioni di classificatori — run `{a.run}`", "",
             f"Coppia `A+B` = classificatore di video (c'e' uno scarico? si'/no) di `A` + modello a finestre (il \"quando\") di `B`, "
             f"ognuno con la sua griglia di iperparametri scelta in CV. Sulla diagonale (`A+A`) i classificatori originali. "
             f"{len(df)} coppie: con tante prove la migliore in CV e' un po' ottimistica anche in CV; il test conferma, non sceglie.", ""]
        for metric, title in (("F1 CV", "F1 della challenge in CV (righe: si'/no, colonne: tempo)"),
                              ("F1 test", "F1 della challenge sul TEST (righe: si'/no, colonne: tempo)")):
            M = df.pivot(index="si/no", columns="tempo", values=metric).reindex(index=order, columns=order)
            best = M.max().max()
            L += [f"## {title}", "", "| si'/no \\ tempo | " + " | ".join(f"`{c}`" for c in order) + " |",
                  "|---|" + "---|" * len(order)]
            for rname, row in M.iterrows():
                cells = [("**" + _fmt(v) + "**") if v == best else _fmt(v) for v in row]
                L.append(f"| `{rname}` | " + " | ".join(cells) + " |")
            L.append("")
        L += ["## Classifica completa per F1 in CV", "", note, ""] + _table_md(df)
        L += ["", "Dettagli di ogni coppia: `classifiers/combinations/<si/no>+<tempo>/runs/" + a.run + "/report.md`.", ""]
        cols = ["classificatore", "F1 CV", "AUC CV", "F1 test", "AUC test"]
    df.drop(columns=[c for c in ("si/no", "tempo") if c in df.columns]).round(4).to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text("\n".join(L), encoding="utf-8")
    print(df[cols].head(15).round(3).to_string(index=False))
    print(f"\nconfronto in {out / (stem + '.md')}")
    return 0


def cmd_run_all(a):
    """Tutto in un colpo, in locale (una GPU): scan -> prepare -> extract -> train-eval -> report."""
    for fn, name in ((cmd_scan, "scan"), (cmd_prepare, "prepare"), (cmd_extract, "extract"), (cmd_train_eval, "train-eval")):
        log.info("=== %s ===", name)
        rc = fn(a)
        if rc not in (0, None):
            log.error("%s terminato con codice %s: mi fermo", name, rc)
            return rc
    return 0


# ---------------------------------------------------------------------------------------------------------------------- parser
def build_parser():
    ap = argparse.ArgumentParser(prog="python -m iwdd", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="controlla ambiente, dataset, modelli in cache")
    p.add_argument("--questions", default=None)
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("questions", help="scrive QUESTIONS.md dal file delle domande")
    p.add_argument("--questions", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_questions)

    p = sub.add_parser("prompts", help="stampa il prompt esatto che riceve Visual-JEV per ogni domanda (audit, senza GPU)")
    p.add_argument("--questions", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--context-mode", default=None, choices=["none", "basic", "rows"])
    p.add_argument("--only", default=None, help="solo questi id, separati da virgola")
    p.set_defaults(fn=cmd_prompts)

    p = sub.add_parser("scan", help="metadati dei video e gruppi di camera (CPU)")
    _common(p)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("prepare", help="immagini evento col preprocessing del progetto (CPU)")
    _common(p)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--reuse-events", default=None, help="cartelle di immagini gia' prodotte da riusare (sola lettura), separate da virgola")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("extract", help="risposte di Visual-JEV alle domande per ogni immagine (GPU)")
    _common(p)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=None, help="domande per blocco (8 GB: 11; A100: >= numero di domande)")
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--context-mode", default=None, choices=["none", "basic", "rows"])
    p.add_argument("--max-minutes", type=float, default=0, help="si ferma (codice 75) dopo questo tempo, finito il video in corso")
    p.add_argument("--limit-videos", type=int, default=None, help="solo i primi N video da fare (prova di tempo)")
    p.add_argument("--force", action="store_true", help="rifa' anche i video gia' estratti / ignora il manifest")
    p.add_argument("--reuse-events", default=None)
    p.add_argument("--wait-for", default=None, help="in parallelo a 'prepare': estrae i video man mano che sono pronti e si ferma "
                                                    "quando questo file esiste (prepare finito) e non resta nulla di pronto")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("groups", help="gruppi di camera su tutti i video scansionati (dopo una scansione divisa in parti)")
    _common(p)
    p.set_defaults(fn=cmd_groups)

    p = sub.add_parser("status", help="quanti video sono scansionati / preparati / estratti")
    _common(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("train-eval", help="addestra sul train (scelte in CV a gruppi) e misura sul test (CPU)")
    _common(p)
    p.add_argument("--train-splits", default="train")
    p.add_argument("--test-splits", default="test", help="'' per non valutare (modello finale su train+test)")
    p.add_argument("--time-col", default=None, choices=[None, "t_start", "t_center", "t_end"])
    p.add_argument("--model-tag", default="default", help="sottocartella di models/ dove salvare i modelli finali")
    p.add_argument("--allow-partial", action="store_true", help="addestra anche se mancano le feature di alcuni video (sconsigliato)")
    p.add_argument("--classifier", default=None, help="classificatore in classifiers/<NOME>/ (default: logistica L2 originale, "
                                                       "uscite nel run); le uscite vanno in classifiers/<NOME>/runs/<run>/")
    p.add_argument("--jobs", type=int, default=None, help="processi per la CV (default: $SLURM_CPUS_PER_TASK, altrimenti 1)")
    p.set_defaults(fn=cmd_train_eval)

    p = sub.add_parser("report", help="rigenera report.md dai numeri gia' calcolati")
    _common(p)
    p.add_argument("--classifier", default=None)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("classifiers", help="elenca i classificatori disponibili (classifiers/*/model.json)")
    p.set_defaults(fn=cmd_classifiers)

    p = sub.add_parser("compare", help="confronta i classificatori addestrati su un run -> classifiers/comparison/<run>.md")
    p.add_argument("--run", default="main")
    p.add_argument("--combinations", action="store_true",
                   help="confronta le coppie A+B (si'/no di A, tempo di B) -> classifiers/comparison/<run>_combinations.md")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("run-all", help="scan, prepare, extract, train-eval, report in locale")
    _common(p)
    for name, kw in (("--workers", dict(type=int, default=4)), ("--reuse-events", dict(default=None)),
                     ("--shard", dict(type=int, default=0)), ("--num-shards", dict(type=int, default=1)),
                     ("--chunk", dict(type=int, default=None)), ("--max-pixels", dict(type=int, default=None)),
                     ("--context-mode", dict(default=None)), ("--max-minutes", dict(type=float, default=0)),
                     ("--limit-videos", dict(type=int, default=None)), ("--train-splits", dict(default="train")),
                     ("--test-splits", dict(default="test")), ("--time-col", dict(default=None)),
                     ("--model-tag", dict(default="default")), ("--classifier", dict(default=None)),
                     ("--jobs", dict(type=int, default=None))):
        p.add_argument(name, **kw)
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.set_defaults(fn=cmd_run_all)
    return ap


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    a = build_parser().parse_args(argv)
    ensure_project_imports()
    return a.fn(a)
