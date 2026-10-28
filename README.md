# ViSFE — Vision-Language Semantic Feature Extraction for Illegal Waste Dumping Detection

Code for the paper **"ViSFE: Vision-Language Semantic Feature Extraction for Illegal Waste Dumping Detection"**
(Galli, Gravina, Moscato, Orlando, Perillo, Picariello, Russo — WasteVision workshop, WACV 2027), submitted to the IWDD 2027 challenge on the
MIVIA-IWDD dataset.

ViSFE does not ask a VLM whether a dumping happens. [Visual-JEV](https://github.com/guanxuyu-sv/Visual-Jev) (Qwen3-VL-4B-Instruct + LoRA adapter)
is used as a *measurer*: for every 3×3 grid of chronologically ordered frames it scores 35 fixed, context-aware questions, and the raw
candidate-answer logits become a numerical feature vector. Two small classifiers then decide **whether** the video contains a dumping
(Event Detection, ED) and **when** (Temporal Localization, TL).

| | Precision | Recall | F1 |
|---|---|---|---|
| Public test set (100 videos, trained on the official train split) | 0.800 | 0.870 | 0.833 |
| Private test set (IWDD 2027, retrained on train + public test) | 0.808 | 0.840 | 0.824 |

Selected configuration: L2 logistic regression for both ED and TL (C = 0.03), temporal offset +2 s.

## Repository layout

```
.
├── main.py, preprocessing.py, preprocessing_pt2.py,   shared project code: Visual-JEV scorer, tracking + 3x3 event grids,
│   decompose_video.py, progress_utils.py               full-frame coverage grids
├── iwdd_pipeline/        training / evaluation pipeline (feature extraction, grouped CV, 7 classifier families, tests)
│   └── README.md         full documentation of the pipeline (in Italian)
├── submission/           self-contained inference package submitted to the contest (test.py, trained ED/TL models)
│   └── README.md         how to run it and what it contains
├── dataset/              (not versioned) MIVIA-IWDD: dataset/{train,test}/{videos,labels}/
└── visual-jev/           (not versioned) clone of https://github.com/guanxuyu-sv/Visual-Jev
```

## Setup

```bash
git clone https://github.com/guanxuyu-sv/Visual-Jev visual-jev          # code of Visual-JEV (Apache-2.0)
# torch / torchvision from the PyTorch index matching your CUDA, then:
pip install -r iwdd_pipeline/requirements.txt
```

Place the MIVIA-IWDD videos and labels under `dataset/` as shown above. The base model (`Qwen/Qwen3-VL-4B-Instruct`) and the adapter
(`guanxuyu/visual-jev-4b-answer-sft`) are downloaded from the Hugging Face Hub. A NVIDIA GPU is needed for feature extraction.

## Reproducing the experiments

```bash
cd iwdd_pipeline
python -m iwdd doctor                                      # check paths, dataset, libraries, GPU, cached models
python -m iwdd run-all --run main                          # scan -> prepare -> extract (GPU) -> train-eval, one GPU
python -m iwdd train-eval --run main --classifier svm      # any of the 7 classifier families on the same features
python -m iwdd compare --run main                          # comparison table (paper Fig. 3)
python -m unittest discover -s tests -v                    # tests, no GPU needed
```

Feature extraction is the expensive step and is done once per run; all classifiers are trained on the same features with the same
camera-grouped cross-validation folds. Extraction can also be split across several GPUs (`--shard`/`--part` options). See
[iwdd_pipeline/README.md](iwdd_pipeline/README.md) for all options.

Classifier families (`iwdd_pipeline/classifiers/`): `logreg` (LR), `logreg_enet` (EN-LR), `svm`, `mlp`, `hist_gb` (GB), `random_forest` (RF),
`decision_tree` (DT).

## Running the contest submission

```bash
cd submission
git clone https://github.com/guanxuyu-sv/Visual-Jev visual-jev
pip install -r requirements.txt
python test.py --videos <folder with .mp4 videos> --results <output folder>
```

It writes one `<video>.txt` per input video with the detection second (or an empty file). Details in [submission/README.md](submission/README.md).
`submission/` is deliberately self-contained (it carries its own copy of the shared scripts and a reduced `iwdd/` package), as the contest
requires a standalone archive.

## Notes

* Metric: contest F1 with valid window `[g − 3 s, g + 10 s]` around the ground-truth second `g`.
* Model and hyper-parameter selection use only the training split (camera-grouped, stratified 5-fold CV repeated 3 times); the test set
  is used for measurement only.
* No shortcut features (duration, resolution, fps, brightness, number of grids) are given to the classifiers.
* External components keep their own licenses: Visual-JEV and Qwen3-VL-4B-Instruct (Apache-2.0).

## License

This repository is released under the [Creative Commons Attribution-NonCommercial 4.0 International](LICENSE) license (CC BY-NC 4.0).
