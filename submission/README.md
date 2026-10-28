# WasteVision — IWDD 2027 contest submission

Illegal waste dumping detection in videos from fixed CCTV cameras. For every test video the method outputs the **second** at which the
dumping happens, or nothing.

```
python test.py --videos <folder with .mp4 videos> --results <output folder>
```

One `<video file name>.txt` per video is written in `--results` (e.g. `vid0041.mp4.txt`): it contains the detection time as an **integer
number of seconds**, or it is **empty** when no dumping is detected. Nothing else is written in that folder (a per-video log goes to
`submission_log.csv` in the working directory).

## Running it

1. Clone Visual-JEV into `visual-jev/` (see the table below) and install `requirements.txt`; a GPU is needed (T4 or better).
2. Run the command above. If a long run is interrupted, add `--resume`.

Internet access is needed once, for the base model weights `Qwen/Qwen3-VL-4B-Instruct` (public, pinned to a fixed revision).
Everything else (adapter, classifiers, code) is in this folder.

## Method

Visual-JEV (a Qwen3-VL-4B vision-language model fine-tuned for multiple-choice questions) is used as a **measurer of 35 visual concepts**;
two small logistic regressions decide *whether* and *when* a dumping happens. No text is generated.

1. **Preprocessing** (CPU, `preprocessing.py`, `preprocessing_pt2.py`). Frames are extracted at 4 fps. For fixed cameras the moving
   subjects are tracked and each event becomes an image made of a 3x3 grid of 9 frames in chronological order, zoomed on the subject;
   for moving cameras the whole frame is used. Stretches of the video that no event grid covers (and videos with no grid at all) get
   full-frame grids (windows of 4 s, stride 2 s), so that every moment of the video is seen by the model.
2. **Visual-JEV as a measurer** (GPU, `main.py`, `iwdd/jev.py`). Qwen3-VL-4B-Instruct (8-bit) with the public Visual-JEV LoRA adapter answers
   the same 35 fixed English questions on every grid (`config/questions.json`: 26 yes/no, 8 claims to be judged
   supported/contradicted/not determined, 1 multiple choice), e.g. *"Is a person carrying a piece of waste?"*, *"Is a vehicle stopped
   with its trunk open?"*. The raw option logits are kept. A short, identical text before every question explains that the image is a
   3x3 chronological grid. The model is only asked to rank candidate answers; it is never prompted to decide on dumping.
3. **Signals** (`iwdd/features.py`). Per grid: the yes-logit of each yes/no question, the log-probability of "supported" for each claim,
   of the first three options for the multiple choice, plus two derived signals (a deposit state and a pick-up state, combining the four
   "first frames / last frames" questions) — 37 numbers.
4. **Video model** ("is there a dumping?"). For each signal: maximum, mean, mean of the 3 best, fraction of grids above 0.5 over the grids
   of the video (149 features) -> L2 logistic regression (C = 0.03). Detection if the score is >= 0.414.
5. **Window model** ("when?"). L2 logistic regression (C = 0.03) on the signals of ONE grid (+ a flag for full-frame grids), trained to
   score whether answering with that grid's time would fall in the valid window. The answer is the end time of the best grid + 2 s,
   rounded to the nearest second and never beyond the end of the video.

Training: the final model is fitted ONCE, without cross-validation, on all the labelled videos available: the 498 usable videos of the 500 of the
MIVIA-IWDD dataset (two videos are excluded by the project configuration). Cross-validation was used only to choose the hyper-parameters
(on 398 videos: 5-fold x 3 repeats, grouped by camera so that videos of the same camera never sit in both training and validation) and the
decision threshold (re-selected on all 498 videos with the same protocol). Training and evaluation code is in `../iwdd_pipeline/`.

Expected performance with the contest metric (time window [g-3 s, g+10 s]): cross-validation on the 498 videos gives F1 0.79 (detection
AUC 0.94). An earlier version of the model, trained on 398 videos only, scored F1 0.83 on the 100 videos held out at that time (95% interval
~0.75-0.91; those videos were looked at while comparing candidate methods, so that figure is slightly optimistic). The final model has no
held-out data left, so the cross-validation figure is the honest estimate. No shortcut feature (duration, resolution, fps, brightness,
number of grids) is given to the models.

## Resources

* **GPU memory**: the 8-bit backbone is ~4.6 GiB and the peak is ~11.3 GiB with the default `--chunk 40` (the setting the model was validated
  with: it reproduces the validation outputs exactly). `--chunk 8` lowers the peak to ~6.3 GiB and is slower, and it changes the GPU
  logits slightly (bf16), which can flip a few borderline decisions: use it only if the GPU runs out of memory.
* **Speed**: GPU time is proportional to the number of grids (about 18 per video on average; ~1.1 s per grid on an A100 with a larger
  chunk, slower on a T4). The CPU preparation of the next videos runs in parallel with the GPU. The method is not real time: it
  analyses whole videos.

## Contents

| path | what |
|---|---|
| `test.py` | entry point required by the contest |
| `infer.py` | inference engine (steps above), robust to unreadable videos (an empty result is written) |
| `main.py`, `preprocessing.py`, `preprocessing_pt2.py`, `decompose_video.py`, `progress_utils.py` | project code, used unmodified |
| `iwdd/`, `config/` | feature extraction code, the 35 questions, settings |
| `visual-jev/` | **not versioned in this repository**: clone https://github.com/guanxuyu-sv/Visual-Jev into `submission/visual-jev/` (Apache-2.0) |
| `models/` | the two classifiers (`video_model.json`, `window_model.json`, `final_config.json`) and the Visual-JEV LoRA adapter (`guanxuyu/visual-jev-4b-answer-sft`) |

External models: `Qwen/Qwen3-VL-4B-Instruct` (Apache-2.0) and the Visual-JEV adapter (Apache-2.0).
