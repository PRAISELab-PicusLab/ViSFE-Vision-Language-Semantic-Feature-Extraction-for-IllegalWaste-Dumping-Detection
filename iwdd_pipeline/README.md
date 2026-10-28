# Pipeline IWDD — Visual-JEV come misuratore di concetti

Rilevamento di scarichi illegali di rifiuti da video di sorveglianza (challenge IWDD 2027) con **Visual-JEV** usato come
*misuratore* di 35 concetti visivi e due piccoli modelli che decidono **se** c'è uno scarico e **quando**: in origine due regressioni
logistiche, ora **sette classificatori a confronto** (cartelle `classifiers/<nome>/`, vedi §6). Tutto gira in locale con una GPU. Il codice sta in `iwdd_pipeline/`; **nessun file originale del
progetto viene modificato** (`main.py`, `preprocessing.py`, `preprocessing_pt2.py`, `decompose_video.py`, `visual-jev/` sono usati
così come sono).

---------------------------------------------------------------------------------------------------------------------------

## 1. Come funziona

```
video ──scan──► metadati, gruppi di camera                                   (CPU)
video ──prepare──► immagini evento (preprocessing del progetto: tracking,     (CPU)
                   zoom sul soggetto, griglia 3×3 da 9 frame) + griglie di copertura
                   a frame intero sui tratti senza griglie (preprocessing_pt2.py)
immagini ──extract──► risposte di Visual-JEV alle 35 domande per ogni griglia (GPU)   ← passo lungo
risposte ──train-eval──► classificatore di video + modello a finestre          (CPU)
                         scelte in CV a gruppi di camera sul TRAIN,
                         misura sul TEST, tabelle, modelli, report.md
```

* **Immagini**: le immagini evento del tuo preprocessing (tempo di una griglia = indice di frame ÷ 4, come nella pipeline originale).
* **Copertura** (`preprocessing_pt2.py`, dentro `prepare`): ogni tratto di almeno `events.fill_gaps_s` secondi (default 2; `null` in
  `config/default.json` la disattiva) non coperto da nessuna griglia evento riceve griglie a **frame intero** (finestre da 4 s, passo 2 s),
  `<video>_fNNN_tA_tB.jpg`; un video senza griglie viene coperto tutto. Senza copertura il 26% degli scarichi dinamici e il 10% degli
  statici cadeva fuori da ogni griglia (e 9 positivi del train non avevano griglie): il "quando" era impossibile da trovare. Il tipo
  di griglia finisce nella colonna `kind` (`event`/`fill`); il **modello a finestre** riceve l'indicatore `is_fill` (frame intero e
  zoom danno risposte diverse), il **classificatore di video** no: aggregato per video misurerebbe quanto del video è "vuoto", cioè la
  durata, una scorciatoia. Il valore di `fill_gaps_s` sta nel manifest delle feature: run con e senza copertura non si mescolano.
  In `events/<video>/coverage.json` restano i buchi riempiti; la cartella del video compare (rinominata da `<video>.prep`) solo quando
  le griglie di copertura ci sono già, così `extract --wait-for` non legge mai una cartella a metà.
* **Domande**: 35, bloccate in `config/questions.json` (le tue 24 + le 11 suggerite). A Visual-JEV va il testo **inglese**; l'italiano
  è solo per leggerle. Elenco completo: `QUESTIONS.md`.
* **Contesto** davanti alla domanda (identico per tutte): 3 righe sulla griglia 3×3, con la riga in alto = primi frame, in basso =
  ultimi (`context_mode: rows`). Alternative: `basic` (senza la frase sulle righe) e `none` (nessun testo).
* **Segnali per griglia** (37): logit di "sì" per le 26 domande sì/no, log-probabilità di "supportata" per le 8 affermazioni, delle
  prime 3 opzioni per la scelta multipla; più 2 segnali derivati da una griglia sola (`dep_state`, `pick_state`) che combinano le quattro
  domande di direzione (primi/ultimi frame).
* **Classificatore di video**: per ogni segnale, max / media / media dei 3 migliori / frazione di griglie sopra 0,5 → regressione logistica L2.
* **Modello a finestre**: regressione logistica L2 sui segnali di UNA griglia; l'etichetta è «rispondere col tempo di questa griglia
  `+ delta` cadrebbe in `[g−3, g+10]`». Risposta = griglia col punteggio massimo, `t_end + delta`, **secondo intero**, non oltre la fine del
  video; si risponde solo se il punteggio di video supera la soglia.
* **Nessuna scorciatoia fra le feature** (scelta del progetto): niente durata, risoluzione, fps, luminosità, notte, numero di griglie,
  loghi. Servono solo alla *baseline di sola scorciatoia* riportata nel report come confronto.
* **Metrica**: F1 della challenge, con la stessa regola di `main.evaluate()` (verificata dai test).

### Cosa decide il programma da solo, e come (sempre sul solo TRAIN)
CV a gruppi di camera (5 fold × 3 ripetizioni; i video della stessa telecamera stanno nello stesso fold):
insieme di feature (`base` o `derived`), aggregazioni, C del classificatore di video, C del modello a finestre, `delta` (−2…+4 s) e soglia di
video. A parità di AUC (entro 0,005) sceglie il modello più semplice. Il **test non influenza nessuna scelta**: si usa solo per misurare.

### La cross-validation, nel dettaglio
* **Tipo**: `StratifiedGroupKFold` di scikit-learn, **5 fold, ripetuta 3 volte** con mescolamenti diversi (seed 0, 1, 2); `iwdd/cv.py`.
  Il numero di fold si riduce da solo se ci sono meno di 5 gruppi o meno di 5 video della classe minoritaria.
* **A gruppi di camera**: i video della stessa telecamera finiscono sempre nello stesso fold. I gruppi li calcola `scan`
  (`iwdd/scan.py`): stesso rapporto d'aspetto e correlazione ≥ 0,95 tra le "firme del fondo" (mediana dei frame, 32×18 grigi).
  *Perché*: la stessa scena compare in più video (misurato: gruppi fino a 7 video, e coppie note come `vid0456`/`vid0071`); se un
  video sta nel train e un altro della stessa camera nella validazione, il modello vedrebbe lo sfondo già noto e la stima verrebbe gonfiata.
* **Stratificata**: ogni fold mantiene la proporzione positivi/negativi (50/50), così AUC e F1 sono confrontabili tra fold.
* **Ripetuta 3 volte**: riduce la dipendenza da un singolo mescolamento. Per ogni video si fa la media delle 3 predizioni
  *out-of-fold* (mai fatte da un modello che lo ha visto in addestramento).
* **A cosa serve** (solo sul TRAIN): scegliere insieme di feature, aggregazioni e C del classificatore di video (massimo AUC out-of-fold,
  a parità il più semplice); scegliere C del modello a finestre; scegliere `delta` e soglia di video massimizzando l'**F1 della challenge**
  sulle predizioni out-of-fold (rilevamento + tempo insieme). Gli stessi fold sono usati per il classificatore, il modello a finestre e la baseline.
* **Limiti**: non è una CV *annidata*: le scelte e la stima "CV" usano gli stessi fold, quindi l'F1 in CV è leggermente ottimistico; il numero
  da citare è quello sul **test**, che non ha influenzato nessuna scelta. Il test può condividere alcune telecamere con il train (misurato:
  14–30 video su 100, a seconda della soglia di somiglianza), quindi anche quello resta un po' ottimistico.

---------------------------------------------------------------------------------------------------------------------------

## 2. Cartelle

```
ViSFE/                                ← cartella del progetto (IWDD_PROJECT)
├── main.py, preprocessing.py, preprocessing_pt2.py, decompose_video.py, progress_utils.py   (usati così come sono, NON modificati)
├── visual-jev/                       ← clone di github.com/guanxuyu-sv/Visual-Jev (non versionato)
├── dataset/{train,test}/{videos,labels}/
├── .venv/                            ← ambiente Python (opzionale)
└── iwdd_pipeline/                    ← QUESTA pipeline
    ├── README.md, QUESTIONS.md, requirements.txt
    ├── config/
    │   ├── questions.json            le 35 domande (id, testo en/it, tipo, opzioni, segnali usati) ← set bloccato
    │   └── default.json              parametri (pixel, contesto, CV, griglie di C e delta, bootstrap)
    ├── iwdd/                         i moduli
    │   ├── paths.py      percorsi e layout dei risultati        ├── scan.py       metadati + gruppi di camera
    │   ├── config.py     lettura config                         ├── events.py     immagini evento (preprocessing)
    │   ├── data.py       video, etichette, shard, sottoinsiemi  ├── jev.py        Visual-JEV: logit grezzi a blocchi
    │   ├── questions.py  domande e contesto                     ├── extract.py    estrazione (resumabile, a shard/parti)
    │   ├── features.py   cache → matrici di feature             ├── models.py     classificatori (LR originale + generici)
    │   ├── cv.py         CV a gruppi, OOF in parallelo          ├── evaluate.py   metrica della challenge, bootstrap
    │   ├── train_eval.py addestramento e valutazione            ├── report.py     report.md
    │   └── cli.py        comandi  (python -m iwdd ...)
    ├── classifiers/                  ← UNA CARTELLA PER CLASSIFICATORE (§6)
    │   ├── <nome>/model.json         tipo di modello e griglie di iperparametri (versionato)
    │   ├── <nome>/runs/<run>/        USCITE di quel modello: report.md, metrics/, tables/, models/
    │   ├── <nome>/logs/              log dei job di addestramento di quel modello
    │   └── comparison/<run>.md|csv   tabella di confronto fra i classificatori
    ├── tests/            test automatici (nessuna GPU)
    └── runs/             ← FEATURE CONDIVISE fra i classificatori ($IWDD_RUNS)
        └── <nome-run>/
            ├── run_config.json       cartelle di immagini da riusare
            ├── meta/                 <video>.json (scansione) e groups.json (gruppi di camera)
            ├── events/<video>/*.jpg  immagini evento
            ├── features/             <video>.parquet (logit grezzi), <video>.windows.parquet (tempi), _manifest.json, timing_shard*.jsonl
            ├── logs/                 errori delle singole fasi
            └── metrics/ tables/ models/ report.md   ← solo se si addestra SENZA --classifier (logistica originale)
```

Variabili d'ambiente che spostano le cartelle: `IWDD_PROJECT`, `IWDD_DATASET`, `IWDD_RUNS`, `IWDD_SCRATCH` (frame temporanei),
`IWDD_CLASSIFIER_OUT` (uscite dei classificatori; la usano i test).
**Un "run"** è un insieme coerente di feature estratte con le stesse domande/contesto/risoluzione. Se cambi anche una sola domanda,
usa un nuovo `--run` (il programma lo impedisce comunque: confronta l'impronta in `features/_manifest.json`). Tutti i classificatori
si addestrano sulle stesse feature del run: l'estrazione (la parte costosa) si fa una volta sola.

---------------------------------------------------------------------------------------------------------------------------

## 3. Comandi

Tutti si lanciano da `iwdd_pipeline/` come `python -m iwdd <comando> [opzioni]`.

| comando | cosa fa | dove |
|---|---|---|
| `doctor` | controlla percorsi, dataset, librerie, GPU, modelli in cache | qualunque |
| `questions` | rigenera `QUESTIONS.md` da `config/questions.json` | qualunque |
| `scan` | metadati reali dei video + gruppi di camera | CPU |
| `prepare` | immagini evento col preprocessing del progetto | CPU |
| `extract` | risposte di Visual-JEV a tutte le domande, per ogni griglia | **GPU** |
| `groups` | gruppi di camera su tutti i video scansionati (dopo una scansione divisa in parti) | qualunque |
| `status` | quanti video sono scansionati / preparati / estratti | qualunque |
| `train-eval` | addestra sul train, misura sul test, scrive tabelle, modelli, `report.md` (`--classifier NOME` per un classificatore) | CPU |
| `report` | rigenera `report.md` dai numeri già calcolati | qualunque |
| `classifiers` | elenca i classificatori disponibili (`classifiers/*/model.json`) | qualunque |
| `compare` | tabella di confronto dei classificatori addestrati su un run → `classifiers/comparison/<run>.md` | qualunque |
| `run-all` | scan → prepare → extract → train-eval, in un colpo (locale, una GPU) | GPU |

**Opzioni comuni** (tutti i comandi che toccano i video): `--run NOME` (default `main`), `--splits train,test` (default), `--limit-train N` e
`--limit-test N` (solo N video stratificati 50/25/25 con seed fisso: per prove), `--config FILE`, `--questions FILE`,
`--part K --num-parts N` (solo la parte K di N: divisione deterministica bilanciata sulla durata dei video, identica per `scan`,
`prepare` ed `extract`; è quella usata dai task del job array).

**Opzioni specifiche**
* `scan`, `prepare`: `--workers N` (processi in parallelo; default 4). `prepare`: `--reuse-events DIR[,DIR]` riusa immagini già prodotte (es.
  `events_frames/`), in sola lettura; la scelta viene ricordata nel run.
* `extract`: `--shard K --num-shards N` (la GPU K elabora lo shard K di N; shard bilanciati sul numero di immagini),
  `--chunk M` (domande per blocco: **8 GB di VRAM → 8**; A100 da 64 GB → ≥ 35), `--max-pixels`, `--context-mode none|basic|rows`,
  `--max-minutes T` (si ferma, codice 75, finito il video in corso), `--limit-videos N` (prova di tempo), `--force` (rifà tutto).
  Riprende da solo: un video è completo se esiste `features/<video>.parquet`.
* `train-eval`: `--train-splits train` e `--test-splits test` (con `--test-splits ""` non valuta), `--time-col t_start|t_center|t_end`,
  `--model-tag NOME` (sottocartella di `models/`), `--allow-partial` (addestra anche con feature mancanti: sconsigliato),
  `--classifier NOME` (uno di `classifiers/`; senza: la logistica L2 originale, uscite nel run), `--jobs N` (processi per la CV:
  default `$SLURM_CPUS_PER_TASK`, altrimenti 1).

### Esempi
```bash
cd iwdd_pipeline
python -m iwdd doctor                                     # prima cosa da fare
python -m iwdd questions                                  # rigenera QUESTIONS.md

# prova locale veloce (24 video di train, 12 di test) con una GPU da 8 GB
python -m iwdd run-all --run prova --limit-train 24 --limit-test 12 --workers 3 --chunk 8

# oppure a passi, e con tutto il dataset
python -m iwdd scan    --run main --workers 8
python -m iwdd prepare --run main --workers 8 --reuse-events ../events_frames
python -m iwdd extract --run main --chunk 8               # su 8 GB; lungo (ore)
python -m iwdd status  --run main
python -m iwdd train-eval --run main
python -m iwdd report  --run main                         # → runs/main/report.md

# altri classificatori sulle stesse feature, e confronto
python -m iwdd classifiers
python -m iwdd train-eval --run main --classifier random_forest --jobs 8   # → classifiers/random_forest/runs/main/
python -m iwdd compare --run main                                          # → classifiers/comparison/main.md

# modello finale per la consegna: addestra su train + test, senza valutare
python -m iwdd train-eval --run main --train-splits train,test --test-splits "" --model-tag final
```

---------------------------------------------------------------------------------------------------------------------------

## 4. Test automatici
```bash
cd iwdd_pipeline && python -m unittest discover -s tests -v
```
43 test, senza GPU (`tests/test_classifiers.py`: ogni classificatore addestra su ogni voce delle sue griglie, end-to-end
di tutti i classificatori con la CV in parallelo, parti del job array disgiunte e complete, confronto; `tests/test_coverage.py`: griglie
di copertura scritte prima che la cartella del video diventi visibile, video senza eventi coperto per intero, copertura disattivabile,
`is_fill` solo nel modello a finestre; estrazione in parallelo a `prepare` e tipo di griglia in `test_extract_fake.py`): set di domande bloccato (35, 26+8+1), regola della metrica identica a `main.evaluate()`, risposta intera e non oltre
il video, shard bilanciati e completi, sottoinsiemi stratificati, matrici di feature, probabilità, e un **end-to-end su feature sintetiche**
(addestramento → valutazione → report → salvataggio e ricarica dei modelli). Scrivono in una cartella temporanea.

---------------------------------------------------------------------------------------------------------------------------

## 5. Esecuzione locale
Serve una GPU NVIDIA. Con 8 GB di VRAM: `--chunk 8` (il default in `config/default.json`, per immagini da ~980 token; con 11 e 35 domande la memoria va in spill: misurato) e tutto
rallenta di 20–60 volte. 

---------------------------------------------------------------------------------------------------------------------------

## 6. Classificatori

Ogni classificatore ha la sua cartella `classifiers/<nome>/` con `model.json` (il tipo di modello e due griglie di iperparametri, una per
il **classificatore di video** e una per il **modello a finestre**, ordinate dal più semplice al più complesso) e, dopo l'addestramento,
le sue uscite (`runs/<run>/`: `report.md`, `metrics/`, `tables/`, `models/`) e i log (`logs/`). Il protocollo è identico per tutti:
stesse feature, stessi fold di CV a gruppi di camera, stessa scelta di feature/aggregazioni/iperparametri (AUC out-of-fold, a parità
entro 0,005 il più semplice), stessa scelta di `delta` e soglia (F1 della challenge out-of-fold), imputazione con la media e
standardizzazione sul solo train, classi bilanciate. Cambia solo lo stimatore.

| cartella | modello | perché |
|---|---|---|
| `logreg` | regressione logistica L2 | il modello originale: riferimento |
| `logreg_enet` | logistica elastic-net / L1 | sparsa: dice quali delle 35 domande servono davvero (si possono togliere le altre e accorciare l'estrazione) |
| `svm` | SVM lineare e RBF | il kernel RBF coglie interazioni fra domande con pochi dati |
| `decision_tree` | albero di decisione | regole leggibili; instabile con pochi dati: limite inferiore dei modelli ad alberi |
| `random_forest` | random forest | non lineare, robusta al rumore e alle feature inutili, poca taratura |
| `hist_gb` | gradient boosting su istogrammi | di solito il più forte su dati tabellari (simile a LightGBM, senza dipendenze in più) |
| `mlp` | rete neurale piccola | combinazioni non lineari "morbide"; piccola e regolarizzata per i pochi dati |

Non inclusi di proposito: k-NN e Naive Bayes (segnali molto correlati fra loro e alta dimensione rispetto ai ~400 video).
**Aggiungere un classificatore**: nuova cartella `classifiers/<nome>/model.json` con un `kind` fra `logreg`, `svm`, `decision_tree`,
`random_forest`, `hist_gb`, `mlp` (costruttori in `iwdd/models.py`) e le due griglie; `submit_all.sh` lo prende da solo.
Il punteggio dell'SVM è la sigmoide della funzione di decisione (niente Platt scaling): serve solo a ordinare e a fissare la soglia.
`compare` ordina per **F1 in CV**: scegliere il modello guardando il test lo renderebbe ottimistico.

---------------------------------------------------------------------------------------------------------------------------

## 7. Come leggere i risultati
* `classifiers/comparison/<run>.md`: i classificatori a confronto (F1 e AUC in CV e sul test, parametri scelti).
* `report.md` (in `classifiers/<nome>/runs/<run>/`): configurazione scelta, F1 su CV e su test con intervalli, dove si perde (non rilevati / tempo sbagliato / falsi allarmi per
  tipo), baseline di sola scorciatoia, segnali più e meno informativi.
* `tables/predictions_test.csv`: una riga per video del test: punteggio di video, risposta (secondi), esito (TP/FP/FN/TN).
* `tables/errors_test.csv`: i video sbagliati con tipo di errore. `tables/window_scores_test.csv`: punteggio di ogni griglia.
* `tables/probabilities_grids.csv`: **tutte le probabilità** di Visual-JEV per ogni griglia e domanda (apribile con Excel), con il tipo di
  griglia (`kind`: `event` = zoom sul soggetto, `fill` = copertura a frame intero).
* `tables/question_stats.csv`: AUC e correlazione di ogni segnale con l'etichetta, su train e test.
* `models/<tag>/final_config.json`: tutto ciò che serve per applicare il modello a un video nuovo (domande, contesto, risoluzione, pesi,
  media/deviazione per la standardizzazione, soglia, delta, regola del tempo). È il ponte verso il `test.py` di consegna (in `../submission/`).

## 8. Problemi frequenti
| messaggio | causa e rimedio |
|---|---|
| `Feature mancanti: N video…` | l'estrazione non è completa: `python -m iwdd status --run X`, rilancia `extract` (riprende) |
| `…estratte con fill_gaps_s=None e ora fill_gaps_s=2.0` | run creato senza copertura (es. `main`) rilanciato con la copertura attiva: usa un nuovo `--run` (es. `cov`) |
| `…set di domande/contesto/risoluzione DIVERSO` | hai cambiato domande o contesto in un run già usato: nuovo `--run`, oppure `--force` per sovrascrivere |
| estrazione lentissima o `OutOfMemory` | `--chunk` troppo alto per la GPU: 11 su 8 GB |
| `No module named 'main'` / `vdm` | `IWDD_PROJECT` non punta alla cartella con `main.py` e `visual-jev/` (esegui `doctor`) |
| `TerminatedWorkerError` (SIGKILL) durante la CV | memoria o CPU insufficienti: ripetere con `--jobs 1` |
| `Classificatore 'X' non trovato` | manca `classifiers/X/model.json` (`python -m iwdd classifiers` elenca quelli disponibili) |
| `PermissionError` rinominando una cartella (Windows) | antivirus: il programma ritenta da solo 3 volte |
