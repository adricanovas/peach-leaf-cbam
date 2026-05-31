# Peach Leaf CBAM Classification

Deep learning pipeline for classifying peach leaf pathologies from cropped leaf images. The project benchmarks 11 CNN backbones with and without a CBAM (Convolutional Block Attention Module) attention mechanism, and evaluates transfer learning (domain shift) from a public dataset to a local field dataset.

---

## Table of Contents

- [Overview](#overview)
- [Classes](#classes)
- [Project Structure](#project-structure)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Running the Experiments](#running-the-experiments)
  - [1. Base models (public dataset, no attention)](#1-base-models-public-dataset-no-attention)
  - [2. CBAM models (public dataset, with attention)](#2-cbam-models-public-dataset-with-attention)
  - [3. Transfer learning — Base (domain shift to local data)](#3-transfer-learning--base-domain-shift-to-local-data)
  - [4. Transfer learning — CBAM (domain shift to local data)](#4-transfer-learning--cbam-domain-shift-to-local-data)
- [Environment Variables](#environment-variables)
- [Outputs](#outputs)
- [Resumability](#resumability)
- [Paper Visualizations](#paper-visualizations)
- [Requirements](#requirements)

---

## Overview

The pipeline is split into four independent training scripts, each runnable from the `code/` directory:

| Script | Dataset | Architecture |
|---|---|---|
| `base_model_train.py` | Public dataset | Standard CNN backbones |
| `cbam_model_train.py` | Public dataset | CNN backbones + CBAM attention |
| `base_tl_model_train.py` | Local field data (transfer learning) | Pre-trained base models fine-tuned |
| `cbam_tl_model_train.py` | Local field data (transfer learning) | Pre-trained CBAM models fine-tuned |

All scripts use repeated stratified k-fold cross-validation (5 splits × 3 repeats), two-stage training (frozen head → fine-tuning), class weighting, and data augmentation.

---

## Classes

**Public dataset** — 6 classes:

Currently empty, include here the public dataset to run the first experiment training. Here is the original distribution:

| Class | Images |
|---|---|
| `healthy` | 951 |
| `bacterial_spot` | 180 |
| `abiotic_stress` | 118 |
| `mite_presence` | 63 |
| `mechanical_stress` | 58 |
| `chewing_insect` | 45 |

**Local dataset** (transfer learning target) — 4 classes:

| Class | Images |
|---|---|
| `healthy` | 107 |
| `mechanical_stress` | 44 |
| `abiotic_stress` | 24 |
| `chewing_insect` | 7 |

---

## Project Structure

```
peach-leaf-cbam/
├── code/
│   ├── base_model_train.py       # Base CNN training on public dataset
│   ├── cbam_model_train.py       # CBAM CNN training on public dataset
│   ├── base_tl_model_train.py    # Transfer learning from base models to local data
│   ├── cbam_tl_model_train.py    # Transfer learning from CBAM models to local data
│   └── utils.py                  # Shared utilities (models, augmentation, metrics)
├── data/
│   ├── public_dataset/
│   │   └── public_data/          # Source training data **(empty, user must upload them here)**
│   └── merged_dataset/
│       └── local_data/           # Target domain data (4 classes, folder-per-class)
├── notebooks/
│   └── paper_visualizations.ipynb  # Figures and tables from results CSVs
├── results/                        # csvs and confusion matrix with the results of the training process
│   ├── base_results/results/
│   ├── cbam_results/results/
│   ├── tl_base/results/
│   └── tl_cbam/results/
├── requirements.txt
└── README.md
```

---

## Dataset Structure

Both datasets must follow a **one folder per class** layout:

```
public_data/
├── healthy/
│   ├── img001.jpg
│   └── ...
├── bacterial_spot/
├── abiotic_stress/
├── mite_presence/
├── mechanical_stress/
└── chewing_insect/

local_data/
├── healthy/
├── mechanical_stress/
├── abiotic_stress/
└── chewing_insect/
```

By default, the scripts expect the data at:
- `../data/public_dataset/public_data` (base and CBAM training)
- `../data/merged_dataset/local_data` (transfer learning)

These paths can be overridden — see [Environment Variables](#environment-variables).

---

## Installation

Python 3.10+ and a CUDA-compatible GPU are recommended.

```bash
git clone https://github.com/adricanovas/peach-leaf-cbam.git
cd peach-leaf-cbam
pip install -r requirements.txt
```

> **Note:** `tensorflow[and-cuda]==2.20.0` is pinned in `requirements.txt`. If running on CPU only, replace it with `tensorflow==2.20.0`.

---

## Running the Experiments

All scripts must be run from the `code/` directory so relative data paths resolve correctly.

```bash
cd code/
```

### 1. Base models (public dataset, no attention)

```bash
python base_model_train.py
```

Trains 11 backbone architectures (MobileNetV2, EfficientNetB0/B3/B5, MobileNetV3Large, DenseNet121, ResNet50/101, InceptionV3, VGG16, VGG19) at two input resolutions (192×192 and 224×224) using 5×3 repeated stratified k-fold CV.

Artifacts are saved to `artifacts/` by default (configurable via `ARTIFACTS_DIR`).

### 2. CBAM models (public dataset, with attention)

```bash
python cbam_model_train.py
```

Same architecture sweep as above, with a CBAM attention block inserted after each backbone. Artifacts saved to `artifacts_cbam/` by default (configurable via `CBAM_ARTIFACTS_DIR`).

### 3. Transfer learning — Base (domain shift to local data)

> Requires Step 1 to have completed: the final `.keras` models must exist in `artifacts/final_model/`.

```bash
python base_tl_model_train.py
```

Loads each best base model and fine-tunes it on the local dataset. Artifacts saved to `artifacts_tl/` (configurable via `TL_ARTIFACTS_DIR`).

### 4. Transfer learning — CBAM (domain shift to local data)

> Requires Step 2 to have completed: the final `.keras` models must exist in `artifacts_cbam/final_model/`.

```bash
python cbam_tl_model_train.py
```

Same as above but starting from CBAM-trained weights. Artifacts saved to `artifacts_cbam_tl/` (configurable via `CBAM_TL_ARTIFACTS_DIR`).

---

## Environment Variables

All artifact paths can be redirected via environment variables, which is useful for running on cloud environments (e.g. Google Colab with Drive mounts):

| Variable | Default | Used by |
|---|---|---|
| `ARTIFACTS_DIR` | `artifacts` | base train, TL base, TL CBAM |
| `CBAM_ARTIFACTS_DIR` | `artifacts_cbam` | CBAM train, TL CBAM |
| `TL_ARTIFACTS_DIR` | `artifacts_tl` | TL base |
| `CBAM_TL_ARTIFACTS_DIR` | `artifacts_cbam_tl` | TL CBAM |
| `PERSISTENT_ARTIFACTS_DIR` | same as `ARTIFACTS_DIR` | base train |
| `PERSISTENT_CBAM_ARTIFACTS_DIR` | same as `CBAM_ARTIFACTS_DIR` | CBAM train |
| `PERSISTENT_TL_ARTIFACTS_DIR` | same as `TL_ARTIFACTS_DIR` | TL base |
| `PERSISTENT_CBAM_TL_ARTIFACTS_DIR` | same as `CBAM_TL_ARTIFACTS_DIR` | TL CBAM |

`PERSISTENT_*` variables allow writing incremental results to a separate location (e.g. a mounted Drive) while keeping intermediate files local.

Example (Colab):

```python
import os
os.environ["ARTIFACTS_DIR"] = "/content/artifacts"
os.environ["PERSISTENT_ARTIFACTS_DIR"] = "/content/drive/MyDrive/peach/artifacts"
```

---

## Outputs

Each training script produces the following under its artifacts directory:

```
artifacts/
├── final_model/
│   ├── <model>_img<size>_final.keras   # Final model per architecture+resolution
│   └── <model>_best_final.keras        # Symlink to best resolution per architecture
├── confusionmatrix/
│   └── <model>_img<size>_cm.png
└── results/
    ├── cv_fold_metrics.csv              # Per-fold metrics for all models
    ├── cv_ranking.csv                   # Cross-validation ranking
    ├── holdout_test_metrics.csv         # Best model holdout results
    ├── holdout_test_metrics_all_models.csv
    ├── best_model_per_architecture.csv
    └── training_progress_state.json     # Resume checkpoint
```

---

## Resumability

All four scripts support **automatic resumption**. If a run is interrupted, re-running the same script will skip already-completed folds and continue from where it left off. Progress is tracked in `training_progress_state.json` inside the results directory.

---

## Paper Visualizations

After all experiments have completed, open the notebook to reproduce the paper figures:

```bash
jupyter notebook notebooks/paper_visualizations.ipynb
```

The notebook reads exclusively from the results CSVs (no model loading required). Update the CSV path variables in **Cell 1** if your artifacts directory differs from the defaults.

---

## Requirements

```
tensorflow[and-cuda]==2.20.0
tensorboard
numpy
pandas
scikit-learn
albumentations
opencv-python-headless
ultralytics
matplotlib
seaborn
```
