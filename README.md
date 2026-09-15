# CIFAR-10 Object Recognition with a Convolutional Neural Network

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13--2.16-orange.svg)](https://www.tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![Test Accuracy](https://img.shields.io/badge/test%20accuracy-82.41%25-brightgreen.svg)](#results)
[![Top-3 Accuracy](https://img.shields.io/badge/top--3%20accuracy-96.43%25-brightgreen.svg)](#results)

A complete, from-scratch **convolutional neural network** that classifies CIFAR-10 images (32×32 RGB) into 10 object categories. The entire pipeline — data loading, splitting, `tf.data` construction, augmentation, model definition, training, evaluation and single-image inference — lives in one self-contained Jupyter notebook with TensorFlow/Keras. No transfer learning, no pretrained weights, no local package imports.

**Pipeline:** `load → split → tf.data → augment → build model → train → evaluate → predict`

Trained on 45,000 images, validated on 5,000 and evaluated on the untouched 10,000-image test set, this model reaches **82.41 % top-1 accuracy** and **96.43 % top-3 accuracy**.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Data Augmentation](#data-augmentation)
- [Callbacks](#callbacks)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Artifacts](#output-artifacts)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Notes and Next Steps](#notes-and-next-steps)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository contains a single Jupyter notebook that builds and trains a CNN for 10-class object recognition on CIFAR-10. It is written to run end to end inside a plain Anaconda environment, and it is deliberately structured so every stage of the pipeline is readable, configurable and re-runnable.

What the notebook implements:

- **A configuration object, not scattered magic numbers.** Every hyperparameter lives on a `TrainConfig` dataclass (`seed`, `batch_size`, `epochs`, `learning_rate`, architecture depth, dropout schedule, augmentation strengths, callback patience, `run_name`, …). Runs are therefore described and serialised by a single object, and each run writes its own `*_config.json`.
- **Reproducible runs.** `set_global_seed()` seeds Python, NumPy and TensorFlow, and sets `PYTHONHASHSEED` / `TF_DETERMINISTIC_OPS`.
- **A real `tf.data` input pipeline.** Images stay `uint8` in memory, are `cache()`d, shuffled, batched, normalised to `[0, 1]` and — for training only — stochastically augmented, all vectorised inside the graph with `AUTOTUNE` parallel calls and prefetching.
- **A 3-stage VGG-style CNN.** Each stage stacks `Conv2D → BatchNormalization → ReLU` blocks, then max-pooling and dropout, before a dense classification head.
- **Regularisation that scales with depth.** L2 weight decay on every conv/dense kernel, batch norm throughout, and a dropout schedule that increases per stage (0.20 → 0.30 → 0.40) plus 0.50 on the dense head.
- **Train-only augmentation.** Random horizontal flip, translation, rotation and zoom are applied through a `tf.keras.Sequential` of preprocessing layers that is active only in the training map function, so validation, test and inference are always deterministic.
- **Production-style callbacks.** Best-checkpoint saving by validation accuracy, early stopping and learning-rate reduction on validation loss, per-epoch CSV logging and `TerminateOnNaN`.
- **A thorough evaluation block.** Top-1 and top-3 accuracy, macro/weighted F1, per-class precision/recall/F1/support, confusion matrices in counts and row-normalised form, and mean cross-entropy computed with a TensorFlow-free NumPy implementation (needed because the model is loaded with `compile=False`).
- **Automatic report figures.** Training curves, both confusion matrices, a per-class accuracy chart and a grid of labelled sample predictions are written to `artifacts/reports/`.
- **Single-image inference.** Classify an image from disk or a CIFAR-10 test index, with a ranked top-k breakdown.
- **A `quick_run` smoke-test mode.** Two epochs on a 2,000-image subset, so the whole pipeline can be verified in a couple of minutes before committing to a long run.

---

## Results

Measured on the full 10,000-image CIFAR-10 test set with the best checkpoint from a 60-epoch run (`artifacts/reports/cifar10_cnn_metrics.json`).

| Metric                     | Value            |
| -------------------------- | ---------------- |
| **Test accuracy (top-1)**  | **82.41 %**      |
| **Test accuracy (top-3)**  | **96.43 %**      |
| Macro F1                   | 0.821            |
| Weighted F1                | 0.821            |
| Test cross-entropy         | 0.514            |
| Test samples               | 10,000           |
| Best validation accuracy   | 82.48 % (epoch 60) |
| Epochs completed           | 60 / 60          |
| Final training accuracy    | 81.17 %          |

![Training and validation curves](docs/images/cifar10_cnn_training_curves.png)

Both curves rise cleanly for the full 60 epochs. The learning rate was reduced on validation-loss plateaus at epoch 24 (1e-3 → 5e-4), epoch 45 (→ 2.5e-4) and epoch 56 (→ 1.25e-4), each step producing a visible jump in both accuracy curves. Training and validation accuracy stay close together throughout, confirming that the combined augmentation + batch-norm + dropout + weight-decay regularisation keeps overfitting in check at this depth.

### Per-class performance

![Per-class accuracy](docs/images/cifar10_cnn_per_class_accuracy.png)

| Class        | Precision | Recall | F1     | Support |
| ------------ | --------- | ------ | ------ | ------- |
| airplane     | 0.833     | 0.859  | 0.846  | 1000    |
| **automobile** | 0.906   | **0.949** | **0.927** | 1000 |
| bird         | 0.853     | 0.710  | 0.775  | 1000    |
| cat          | 0.770     | **0.645** | 0.702 | 1000  |
| deer         | 0.868     | 0.737  | 0.797  | 1000    |
| dog          | 0.843     | 0.662  | 0.742  | 1000    |
| frog         | 0.692     | 0.935  | 0.795  | 1000    |
| horse        | 0.805     | 0.902  | 0.851  | 1000    |
| ship         | 0.869     | 0.918  | 0.893  | 1000    |
| truck        | 0.848     | 0.924  | 0.884  | 1000    |

### Confusion matrices

![Confusion matrix (counts)](docs/images/cifar10_cnn_confusion_matrix.png)

![Confusion matrix (row-normalised)](docs/images/cifar10_cnn_confusion_matrix_normalized.png)

The error structure is exactly what a from-scratch CIFAR-10 model is expected to show:

- **Easiest classes.** `automobile` (0.949 recall), `frog` (0.935), `truck` (0.924), `ship` (0.918) and `horse` (0.902) are the classes the model separates most reliably. The vehicles and vessels (`automobile`, `truck`, `ship`) share rigid, geometric silhouettes; `frog` has an unusually uniform green texture that survives at 32×32; `horse` has a distinctive elongated body and leg pattern.
- **Hardest classes.** `cat` (0.645), `dog` (0.662), `bird` (0.710) and `deer` (0.737) carry the recall cost. The dominant confusions are `cat ↔ dog` (78 cats read as dogs, 105 dogs read as cats), `cat → frog` (97) and `bird → frog` (104) / `deer → frog` (108), all driven by shared fur/feather texture and natural-background context at 32×32 resolution.
- **Asymmetric precision.** `frog` has the lowest precision (0.692) even though it has the highest recall (0.935): it is a frequent false-positive home for cats, deer and birds. Symmetrically, `ship` and `airplane` are sometimes predicted as each other (46→ship, 43→airplane) because both sit on uniform blue backgrounds.

### Sample predictions

![Sample test predictions](docs/images/cifar10_cnn_sample_predictions.png)

A 4×4 grid of test images with the predicted class and confidence, coloured green when correct and red when wrong, plus the ground-truth label beneath each.

---

## Dataset

**CIFAR-10** — 60,000 labelled 32×32 RGB images across 10 mutually exclusive classes, split 50,000 train / 10,000 test by the official benchmark. Keras downloads and caches it automatically on first run (`~/.keras/datasets/`), so no manual data step is required.

The notebook keeps every image as `uint8` (the whole training set fits in roughly 150 MB of RAM in that form) and normalises to `float32` in `[0, 1]` only after batching, inside the `tf.data` graph.

| Index | Class      | Index | Class |
| ----- | ---------- | ----- | ----- |
| 0     | airplane   | 5     | dog   |
| 1     | automobile | 6     | frog  |
| 2     | bird       | 7     | horse |
| 3     | cat        | 8     | ship  |
| 4     | deer       | 9     | truck |

**Splits used by this run**

| Split      | Size    | Source                                                             |
| ---------- | ------- | ------------------------------------------------------------------ |
| Train      | 45,000  | Official train set, 90 % after a seeded 10 % validation hold-out    |
| Validation | 5,000   | Official train set, 10 % held out by a seeded permutation           |
| Test       | 10,000  | Official test set, never seen during training or model selection    |

The validation hold-out is drawn with `np.random.default_rng(seed)` so the same images land in validation on every run with `seed = 42`. `split_train_validation()` always leaves at least one training example and validates that `validation_split ∈ [0, 1)`.

---

## Notebook Walkthrough

The notebook is organised into twelve numbered sections, designed to be executed top to bottom. Sections 1–6 are cheap (they define configuration, data and model); section 7 is the training run.

| §  | Section | Contents |
| -- | ------- | -------- |
| 1  | Environment setup | Imports, library version printout, inline plotting |
| 2  | Configuration | Constants, `CLASS_NAMES`, `artifacts/` layout, the `TrainConfig` dataclass |
| 3  | Reproducibility and runtime | `set_global_seed`, UTF-8 console fix, GPU memory growth |
| 4  | Data pipeline | Loading, train/val split, sanity-check figure, augmentation stack, `tf.data` builders |
| 5  | Model architecture | Stage builder, `build_model`, `count_parameters`, `load_trained_model` |
| 6  | Callbacks | Checkpoint, early stopping, LR reduction, CSV logger, NaN termination |
| 7  | Training | `train()` end-to-end, plus a reloadable history table |
| 8  | Metrics (TensorFlow-free) | Top-k accuracy, NumPy cross-entropy, confusion counts, summary, text formatting |
| 9  | Report figures | Training curves, confusion matrices, per-class accuracy, sample predictions |
| 10 | Evaluation | `run_evaluation()` on the test set, figure/metric persistence, reload of saved predictions |
| 11 | Single-image prediction | Classify from disk or by test index, with ranked top-k output |
| 12 | Notes and next steps | CPU/GPU guidance, smoke test, artifact locations, ablation switches |

Two details worth calling out:

- **Section 8 deliberately avoids TensorFlow.** Because the saved model is loaded with `compile=False` (evaluation and inference only need the forward pass, and this avoids failures when custom optimisers or legacy configs are missing), the loss/metric helpers are plain NumPy so they can be reasoned about and tested independently.
- **Section 3 fixes a Windows-specific papercut.** Keras prints box-drawing characters in its progress output; `use_utf8_console()` reconfigures stdout/stderr to UTF-8 so these never raise `UnicodeEncodeError` on a cp1252 console.

---

## Model Architecture

A `tf.keras.Sequential` built stage by stage. Each convolutional stage is `Conv2D → BatchNormalization → ReLU` repeated `conv_blocks_per_stage` times, followed by max-pooling and dropout. Convolutions use 3×3 kernels, `same` padding and `use_bias=False` (the following batch-norm layer supplies the bias). Every conv and dense kernel carries L2 weight decay (`1e-4`).

| Stage | Layer(s)                                                  | Output shape |
| ----- | --------------------------------------------------------- | ------------ |
| Input | —                                                         | 32×32×3      |
| 1     | 2 × (Conv2D 32, 3×3 → BatchNorm → ReLU) → MaxPool2D 2×2 → Dropout 0.20 | 16×16×32 |
| 2     | 2 × (Conv2D 64, 3×3 → BatchNorm → ReLU) → MaxPool2D 2×2 → Dropout 0.30 | 8×8×64   |
| 3     | 2 × (Conv2D 128, 3×3 → BatchNorm → ReLU) → MaxPool2D 2×2 → Dropout 0.40 | 4×4×128 |
| Head  | Flatten → Dense 256 (no bias) → BatchNorm → ReLU → Dropout 0.50 → Dense 10 Softmax | 10 |

Roughly **0.8 M parameters** in total — the notebook prints the exact trainable/total counts with `count_parameters()` when the model is built.

**Design rationale**

- **Depth in stages, not one long stack.** Three stages let the network learn edges and colour blobs at 32×32, textures and parts at 16×16, and object-level structure at 8×8 and 4×4, halving spatial resolution each time.
- **Two conv blocks per stage.** Stacking two 3×3 convolutions before pooling gives the effective receptive field of a 5×5 kernel with fewer parameters and an extra nonlinearity, and `same` padding preserves spatial dimensions within a stage.
- **Filter depth doubles as resolution halves** (32 → 64 → 128), keeping the information bottleneck roughly constant while the representation grows more abstract.
- **Batch normalisation after every conv and on the dense head** makes the deeper stack trainable at a higher learning rate and stabilises convergence — visible in the smooth, non-spiking loss curves.
- **An increasing dropout schedule** (0.20 → 0.30 → 0.40) targets regularisation where the feature maps are largest and most prone to overfitting; 0.50 on the dense head guards the final 256-unit layer.
- **L2 weight decay + dropout + augmentation** combine to keep validation accuracy tracking training accuracy instead of diverging.
- **Softmax output** gives a proper probability distribution over the 10 classes, which is what makes the top-k and confidence reporting in section 11 meaningful.

The architecture is fully driven by `TrainConfig`: `conv_filters`, `conv_blocks_per_stage`, `dense_units`, `dropout_conv` and `dropout_dense` can be changed without touching the builder code. `_stage_dropouts()` pads the dropout tuple with its last value if it is shorter than the filter list, and `build_model()` raises a clear `ValueError` if `conv_filters` is empty.

---

## Training Configuration

Every tunable value lives on `TrainConfig`, and each run serialises itself to `artifacts/reports/<run_name>_config.json`. The values behind the published results:

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| `seed` | 42 | Seeds Python, NumPy and TensorFlow |
| `batch_size` | 64 | |
| `epochs` | 60 | Completed in full |
| `learning_rate` | 1e-3 | Adam, reduced automatically on plateaus |
| `validation_split` | 0.1 | 5,000 held-out images |
| `shuffle_buffer` | 10,000 | |
| `conv_filters` | (32, 64, 128) | One entry per stage |
| `conv_blocks_per_stage` | 2 | |
| `dense_units` | 256 | |
| `dropout_conv` | (0.20, 0.30, 0.40) | Per stage |
| `dropout_dense` | 0.50 | |
| `weight_decay` | 1e-4 | L2 on conv and dense kernels |
| `use_augmentation` | True | Set `False` for a no-augmentation baseline |
| `early_stopping_patience` | 15 | On `val_loss` |
| `reduce_lr_patience` | 5 | On `val_loss` |
| `reduce_lr_factor` | 0.5 | Halves the LR at each plateau |
| `min_learning_rate` | 1e-6 | Floor for LR reduction |
| `run_name` | `cifar10_cnn` | Prefix for every artifact filename |
| `quick_run` | False | `True` ⇒ 2 epochs on a 2,000-image subset |

**Loss and optimiser.** Sparse categorical cross-entropy over integer class indices (the labels are `(N,)` integer arrays, not one-hot), optimised with Adam at 1e-3 and tracked by `sparse_categorical_accuracy`.

**Learning-rate schedule actually observed.** 1e-3 for epochs 1–24, 5e-4 for 25–45, 2.5e-4 for 46–56, 1.25e-4 for 57–60 — three automatic reductions, each followed by an immediate improvement in validation accuracy.

| Parameter        | Value |
| ---------------- | ----- |
| Loss function    | `sparse_categorical_crossentropy` |
| Optimizer        | Adam (initial LR 1e-3, plateau-reduced) |
| Metric           | `sparse_categorical_accuracy` |
| Input shape      | 32×32×3 |
| Output classes   | 10 |

---

## Data Augmentation

Augmentation is applied **only inside the training map function**, through a `tf.keras.Sequential` of Keras preprocessing layers. Validation, test and inference go through the deterministic `build_eval_dataset()` path, so reported metrics are never computed on augmented data.

| Transform | Setting | Keras layer |
| --------- | ------- | ----------- |
| Horizontal flip | on | `RandomFlip(mode="horizontal")` |
| Translation | 12.5 % of height/width | `RandomTranslation`, `fill_mode="reflect"` |
| Rotation | ±10 % of 2π | `RandomRotation`, `fill_mode="reflect"` |
| Zoom | 10 % | `RandomZoom`, `fill_mode="reflect"` |

Each transform is either enabled or omitted entirely, and `build_augmentation()` returns an *empty* `tf.keras.Sequential` when augmentation is disabled — so the same code path serves both the augmented and the no-augmentation baseline. `fill_mode="reflect"` avoids introducing black borders that the network could learn to key on.

For the horizontal flip this is a genuinely safe augmentation for CIFAR-10 (a mirrored truck is still a truck), while the small translation/rotation/zoom factors teach the model to tolerate the mild framing and scale variation present between object instances. Set `CONFIG.use_augmentation = False` in section 2 to reproduce a no-augmentation ablation.

---

## Callbacks

Section 6 splits the two monitoring responsibilities deliberately: the artifact of record is chosen by the metric we care about, while the *scheduling* decisions use a smoother signal.

| Callback | Monitor | Mode | Configuration |
| -------- | ------- | ---- | ------------- |
| `ModelCheckpoint` | `val_sparse_categorical_accuracy` | max | `save_best_only=True` → `artifacts/models/<run_name>.keras` |
| `EarlyStopping` | `val_loss` | min | patience 15, `min_delta=1e-4` |
| `ReduceLROnPlateau` | `val_loss` | min | factor 0.5, patience 5, `min_lr=1e-6`, `cooldown=1` |
| `CSVLogger` | — | — | Per-epoch log → `artifacts/logs/<run_name>.csv` |
| `TerminateOnNaN` | — | — | Aborts immediately on a NaN loss |

**Why two different monitors**

- **Accuracy defines the artifact.** The model that is saved is the one with the best validation *accuracy*, because that is the quantity being reported and compared.
- **Loss drives stopping and scheduling.** Validation loss is a smoother, lower-variance signal than accuracy, so it gives a less noisy early-stopping decision and less frequent spurious LR reductions. `min_delta=1e-4` ignores the sub-noise improvements that would otherwise reset the patience counter.

`restore_best_weights=False` is intentional: the best weights are already persisted to disk by `ModelCheckpoint`, so the in-memory model can keep improving (or be discarded) without the two mechanisms fighting over the same weights. Early stopping did not trigger in the published run — all 60 epochs completed, with the best checkpoint landing on the final epoch.

---

## Evaluation Metrics

`run_evaluation()` loads the best checkpoint, predicts on the untouched test set and computes the full metric set. The model is loaded with `compile=False`, so the loss is recomputed in NumPy by `sparse_categorical_crossentropy_loss()` — deliberately TensorFlow-free so it can be reasoned about (and tested) with plain arrays.

| Metric | Function | Reported value |
| ------ | -------- | -------------- |
| Top-1 accuracy | `sklearn.metrics.accuracy_score` | 82.41 % |
| Top-3 accuracy | `top_k_accuracy(..., k=3)` | 96.43 % |
| Macro F1 | `classification_report` (macro avg) | 0.821 |
| Weighted F1 | `classification_report` (weighted avg) | 0.821 |
| Per-class precision / recall / F1 / support | `classification_report` (per class) | see [Results](#results) |
| Mean cross-entropy | `sparse_categorical_crossentropy_loss` | 0.514 |
| Confusion matrix (counts) | `sklearn.metrics.confusion_matrix` | rows = true, columns = predicted |
| Confusion matrix (row-normalised) | row sums | per-class recall on the diagonal |

Supporting helpers:

- `top_k_accuracy()` validates that the probability array is 2-D and that `k ≥ 1`, clipping `k` to the number of classes.
- `sparse_categorical_crossentropy_loss()` clips probabilities at `1e-7` before taking the log, so a zero probability for the true class yields a large finite penalty instead of `inf`.
- `format_confusion_matrix_counts()` renders the matrix as a width-aligned text block, so it stays readable in the console where a plot is not wanted.

### Inference output

Section 11 classifies one image — either from a file (`load_image_batch`, which accepts `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp` and resizes to 32×32) or from the test set by index (`load_sample_batch`, which also returns the ground-truth label). A `.npz` of raw labels, predictions and probabilities is written alongside the figures, so predictions can be re-analysed later without re-running the model:

```python
stored = np.load("artifacts/reports/cifar10_cnn_predictions.npz")
print(stored.files)                 # ['labels', 'predictions', 'probabilities']
```

`rank_predictions()` returns the top-k `(class_name, probability)` pairs, and the result dictionary reports the source, the winning class, its confidence, the ranked top-k and (when known) the true label.

---

## Output Artifacts

Everything the pipeline produces lands under `artifacts/`, with filenames derived from `CONFIG.run_name` so multiple experiments can coexist simply by changing that value. The directory is created on demand by `ensure_directories()`.

```
artifacts/
├── models/
│   └── cifar10_cnn.keras                              # best checkpoint (by val accuracy)
├── logs/
│   └── cifar10_cnn.csv                                # per-epoch loss / accuracy / LR
└── reports/
    ├── cifar10_cnn_config.json                        # the exact TrainConfig of the run
    ├── cifar10_cnn_history.json                       # per-epoch history + run summary
    ├── cifar10_cnn_metrics.json                       # summary metrics + confusion matrix
    ├── cifar10_cnn_predictions.npz                    # labels, predictions, probabilities
    ├── cifar10_cnn_training_curves.png
    ├── cifar10_cnn_confusion_matrix.png
    ├── cifar10_cnn_confusion_matrix_normalized.png
    ├── cifar10_cnn_per_class_accuracy.png
    └── cifar10_cnn_sample_predictions.png
```

`artifacts/` is listed in `.gitignore`: the checkpoint and figures are large binaries that are regenerated by a training run. The published figures in this README are committed separately under `docs/images/` so the results are visible without cloning and training.

---

## Project Structure

```
cnn-object-recognition/
├── CNN Object Detection.ipynb    # The whole pipeline: config → data → model → train → evaluate → predict
├── README.md                     # This file
├── requirements.txt              # Pinned dependency ranges
├── conftest.py                   # Puts the project root on sys.path for pytest runs
├── .gitignore                    # Excludes caches, virtualenvs and generated artifacts
├── docs/
│   └── images/                   # Report figures committed for the README
└── artifacts/                    # Generated by a training run (git-ignored)
    ├── logs/
    ├── models/
    └── reports/
```

There is no importable Python package to install: the notebook is self-contained by design, and every function it uses is defined in the notebook itself. `conftest.py` exists only so that a `pytest` invocation from the repository root resolves imports against the project root rather than a `tests/` subdirectory.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/cnn-object-recognition.git
cd cnn-object-recognition
```

### 2. Create an environment (recommended)

```bash
conda create -n cifar10 python=3.10 -y
conda activate cifar10
```

or with plain `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
```

### 3. Install dependencies

Either install from the pinned requirements file:

```bash
pip install -r requirements.txt
```

or use the conda one-liner from the notebook header:

```bash
conda install -c conda-forge tensorflow numpy pandas scikit-learn matplotlib seaborn jupyter
```

**Dependencies**

| Package | Version range | Role |
| ------- | ------------- | ---- |
| `tensorflow` | `>=2.13,<2.17` | Model, training, `tf.data` |
| `numpy` | `>=1.23,<2.0` | Array maths, NumPy metrics |
| `pandas` | `>=1.5` | History table rendering |
| `scikit-learn` | `>=1.2` | Classification report, confusion matrix |
| `matplotlib` | `>=3.6` | All figures |
| `seaborn` | `>=0.12` | Confusion-matrix heatmaps, palettes |
| `jupyter` | `>=1.0` | Running the notebook |
| `pytest` | `>=7.0` | Test runner |

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open `CNN Object Detection.ipynb` and run the cells in order.

> **Note on TensorFlow 2.15.** On import you will see oneDNN / deprecation warnings. These are benign and expected — the notebook calls this out explicitly in section 1.

---

## Usage

The notebook is designed to run top to bottom. Sections 1–6 are cheap; section 7 trains.

### Smoke test first

Before committing to a full run, flip the flag in section 2 (or the cell in section 7.1) to verify the whole pipeline and every artifact:

```python
CONFIG.quick_run = True     # 2 epochs on a 2,000-image subset — a couple of minutes on CPU
```

Leave it `False` for the full 60-epoch run that produced the results above.

### Full training run

```python
report = train(CONFIG)
```

`train()` seeds the run, prints the device, builds the datasets and the model, fits with all callbacks, saves the training-curve figure, writes `<run_name>_config.json` and `<run_name>_history.json`, and returns a summary dict:

```python
{
    "run_name": "cifar10_cnn",
    "seed": 42,
    "device": "CPU only (no GPU detected)",
    "epochs_completed": 60,
    "best_epoch": 60,
    "best_val_accuracy": 0.8248,
    "final_train_accuracy": 0.8117,
    "duration_seconds": ...,
    "parameters": {"trainable": ..., "non_trainable": ..., "total": ...},
    ...
}
```

### Evaluation

Section 10.1 requires a trained checkpoint:

```python
evaluation = run_evaluation(CONFIG)
evaluation["summary"]["accuracy"], evaluation["summary"]["top_3_accuracy"]
# (0.8241, 0.9643)
```

This writes all four figures, `cifar10_cnn_metrics.json` and `cifar10_cnn_predictions.npz`, and prints a per-class table plus the confusion matrix to the console.

### Classify a single image

By test-set index (change `SAMPLE_INDEX` to any value in `[0, 9999]`):

```python
SAMPLE_INDEX = 7
example = run_prediction(CONFIG, index=SAMPLE_INDEX, top_k=5)
```

Or from a file on disk (resized to 32×32 automatically):

```python
IMAGE_PATH = r"C:\path\to\your\image.jpg"
file_example = run_prediction(CONFIG, image_path=IMAGE_PATH, top_k=5)
```

Both return a dictionary with the source, the predicted class, its confidence, the ranked top-k and the true label when one is known.

### Reload the model yourself

```python
model = tf.keras.models.load_model("artifacts/models/cifar10_cnn.keras", compile=False)
probabilities = model.predict(image_batch, verbose=0)   # (1, 10)
```

`compile=False` is the notebook's own choice at evaluation time — only the forward pass is needed there.

---

## Reproducibility

- **Seeding.** `set_global_seed(CONFIG.seed)` seeds Python (`random`), NumPy and TensorFlow, and sets `PYTHONHASHSEED` and `TF_DETERMINISTIC_OPS`. The validation split uses `np.random.default_rng(seed)`, so the same 5,000 images are held out on every run with the same seed.
- **Honest caveat on determinism.** Exact bit-for-bit reproducibility on GPU additionally requires disabling cuDNN autotuning, which costs performance. Seeding the standard generators makes runs *comparable* across attempts; it does not guarantee identical weights. The notebook states this explicitly rather than pretending otherwise.
- **Self-describing runs.** Each run writes its own `*_config.json`, so any published metric can be traced back to the exact configuration that produced it.
- **Switching runs.** Change `CONFIG.run_name` to keep multiple configurations side by side — every artifact filename is derived from it, so nothing is overwritten.

---

## Notes and Next Steps

### Operational notes

- **CPU vs GPU.** The default `TrainConfig` is tuned for CPU-only training: 60 epochs at batch size 64. On a GPU the same settings finish far sooner, and `batch_size` and `epochs` can be raised.
- **Where results land.** See [Output Artifacts](#output-artifacts). Everything is under `artifacts/`.
- **Augmentation ablation.** Set `CONFIG.use_augmentation = False` to reproduce a no-augmentation baseline against the same splits.
- **Reproducing a clean run.** Delete `artifacts/` before a fresh run if you want to be certain no stale checkpoint is picked up — `run_evaluation()` raises `FileNotFoundError` with a clear message when no checkpoint exists.

### Where the remaining error is

The confusion matrices show the model is already strong on rigid, distinctive classes and that nearly all of its residual error is concentrated in the fine-grained natural classes — `cat`, `dog`, `bird`, `deer` — and in the consistent `cat ↔ dog` / `animal → frog` pairs. Improvements should therefore target those classes rather than the architecture wholesale.

### Plausible next steps

- **Higher input resolution** (upsampled or via `RandomResizedCrop`-style preprocessing) to give fine-grained classes more pixels to work with.
- **Cutout / random erasing** on top of the existing geometric augmentation — a cheap, well-established accuracy gain on CIFAR-10, and an easy addition to the existing `tf.keras.Sequential` stack.
- **Stochastic depth or residual connections** to allow a deeper backbone without the degradation that a plain deep stack suffers.
- **MixUp / label smoothing** to soften the confident wrong predictions that show up as off-diagonal mass in the confusion matrix.
- **Test-time augmentation** (average predictions over flips) for a small, essentially free accuracy bump at inference.
- **Class-balanced sampling or focal loss** if the `frog` precision / `cat`+`dog` recall asymmetry is worth trading for.
- **A proper `tests/` suite** around the TensorFlow-free helpers in section 8 (`top_k_accuracy`, `sparse_categorical_crossentropy_loss`, `split_train_validation` boundary conditions) — they are pure NumPy and trivially unit-testable; `pytest` is already in `requirements.txt` and `conftest.py` already handles the import path.

---

## Acknowledgements

- **CIFAR-10** — Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009); distributed by the University of Toronto and loaded here via `tf.keras.datasets.cifar10`.
- **TensorFlow / Keras** — model, `tf.data` pipeline, preprocessing layers and callbacks.
- **scikit-learn** — classification report and confusion matrix.
- **seaborn** and **matplotlib** — all report figures.
