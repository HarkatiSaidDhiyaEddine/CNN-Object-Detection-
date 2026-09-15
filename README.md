# CIFAR-10 Object Recognition with a Convolutional Neural Network

A from-scratch convolutional neural network that classifies **CIFAR-10** images — 32×32 RGB photographs across 10 object categories — built with **TensorFlow / Keras**. No transfer learning, no pre-trained weights: the model is trained end to end.

The project is packaged as a small, installable Python library (`cifar10_cnn/`) with a command-line entry point per stage (**train · evaluate · predict**), a full `tf.data` input pipeline, on-the-fly image augmentation, callback-driven training, and a TensorFlow-free metrics layer so the analysis code can be unit-tested cheaply. It also ships as a single self-contained notebook for anyone who prefers to run everything interactively.

| | |
|---|---|
| **Task** | 10-class image classification |
| **Dataset** | CIFAR-10 (60,000 images, 50k train / 10k test) |
| **Input** | 32 × 32 × 3 RGB |
| **Model** | 3-stage CNN, ~815k trainable parameters |
| **Framework** | TensorFlow 2.15 / Keras |
| **Test suite** | 68 tests, all passing |

---

## Table of Contents

- [Highlights](#highlights)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command line](#command-line)
  - [Python API](#python-api)
  - [Notebook](#notebook)
- [Output Artifacts](#output-artifacts)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Results](#results)
- [Design Notes](#design-notes)
- [Roadmap](#roadmap)
- [License](#license)

---

## Highlights

- **From-scratch CNN** — three convolutional stages, each stacking two `Conv2D → BatchNorm → ReLU` blocks followed by max-pooling and dropout.
- **Regularised for CPU training** — batch normalisation, L2 weight decay, a per-stage dropout schedule that grows with depth, and stochastic augmentation (flip, translate, rotate, zoom).
- **Efficient input pipeline** — images stay `uint8` and are `cache()`d in that form (~150 MB for the whole training set); normalisation and augmentation run *after* batching so the preprocessing layers execute vectorised.
- **Best-checkpoint selection** — `ModelCheckpoint` tracks validation accuracy (the metric of record); `EarlyStopping` and `ReduceLROnPlateau` track the smoother validation loss.
- **Rich evaluation** — accuracy, top-3 accuracy, macro/weighted F1, per-class precision/recall/F1, a confusion matrix (counts + row-normalised), a per-class accuracy chart, and a labelled sample-prediction grid.
- **Reproducible** — one seed drives Python, NumPy and TensorFlow; the full config is serialised next to every run.
- **Tested** — 68 unit tests covering config, data, model, metrics, augmentation, callbacks, plots, prediction, seeding and runtime.

---

## Dataset

**CIFAR-10** — 60,000 labelled 32×32 colour images, split into 50,000 training and 10,000 test images across 10 mutually exclusive classes (6,000 images per class).

| Index | Class        | Index | Class    |
|-------|--------------|-------|----------|
| 0     | `airplane`   | 5     | `dog`    |
| 1     | `automobile` | 6     | `frog`   |
| 2     | `bird`       | 7     | `horse`  |
| 3     | `cat`        | 8     | `ship`   |
| 4     | `deer`       | 9     | `truck`  |

The dataset is downloaded automatically by Keras on first use and cached under `~/.keras/datasets/`:

```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

A further **10% of the training split** is held out as a validation set (`validation_split=0.1`), leaving the 10,000-image test set untouched until final evaluation.

Pixel values are stored as `uint8` in `[0, 255]` and scaled to `float32` in `[0, 1]` by dividing by `255.0`.

---

## Model Architecture

A `tf.keras.Sequential` model built layer-by-layer from a `TrainConfig`. It is three convolutional stages deep; each stage repeats a `Conv2D → BatchNormalization → ReLU` block, then max-pools and applies dropout.

| Stage | Layers |
|-------|--------|
| **Input** | `Input(32, 32, 3)` |
| **Stage 1** | `Conv2D(32, 3×3, same)` → `BN` → `ReLU` ×2 → `MaxPool2D(2)` → `Dropout(0.20)` |
| **Stage 2** | `Conv2D(64, 3×3, same)` → `BN` → `ReLU` ×2 → `MaxPool2D(2)` → `Dropout(0.30)` |
| **Stage 3** | `Conv2D(128, 3×3, same)` → `BN` → `ReLU` ×2 → `MaxPool2D(2)` → `Dropout(0.40)` |
| **Head** | `Flatten` → `Dense(256)` → `BN` → `ReLU` → `Dropout(0.50)` → `Dense(10, softmax)` |

```
Input (32, 32, 3)
  └─ Stage 1: conv×2 @32 → pool → 16×16×32 → dropout 0.20
  └─ Stage 2: conv×2 @64 → pool →  8×8×64  → dropout 0.30
  └─ Stage 3: conv×2 @128→ pool →  4×4×128 → dropout 0.40
  └─ Flatten → Dense(256) → BN → ReLU → Dropout 0.50
  └─ Dense(10, softmax)
```

**Parameter counts** (default configuration):

| | Count |
|---|---|
| Trainable | **814,826** |
| Non-trainable (batch-norm statistics) | 1,408 |
| **Total** | **816,234** |

**Design rationale:**

- **Progressively deeper channels (32 → 64 → 128)** keep a growing information bottleneck: spatial resolution halves at each pool while channel depth doubles, so the network learns abstract features before the classification head.
- **Batch normalisation** after every convolution makes the deeper stack trainable at a higher learning rate and stabilises convergence.
- **`use_bias=False` + batch norm** is redundant to bias on the conv layers, so the bias is dropped — BN's shift term replaces it.
- **L2 weight decay** (`1e-4`) is applied to convolution and the dense head as a second regulariser alongside dropout.
- **Dropout that increases with depth** (0.20 → 0.40) targets the layers whose feature maps are most prone to memorising the training set; the head uses a heavier 0.50.
- **Small 3×3 kernels with `same` padding** preserve spatial size within a stage and keep the parameter count modest.
- **Softmax output** yields a probability distribution over the 10 classes, so the argmax is the predicted label.

---

## Training Configuration

| Setting | Value |
|---|---|
| Loss | `sparse_categorical_crossentropy` |
| Optimizer | `Adam` (learning rate `1e-3`) |
| Metric | `sparse_categorical_accuracy` |
| Batch size | 64 |
| Epochs | 60 (with early stopping) |
| Validation split | 0.1 |
| Checkpoint monitor | `val_sparse_categorical_accuracy` (max) |
| Early-stopping monitor | `val_loss` (patience 15) |
| LR schedule | `ReduceLROnPlateau` on `val_loss` (×0.5, patience 5, min 1e-6) |
| Weight decay | L2, `1e-4` |

Sparse categorical cross-entropy is used because the labels are integer class indices rather than one-hot vectors.

Callbacks registered for every run:

| Callback | Purpose |
|---|---|
| `ModelCheckpoint` | Persists the best model by validation accuracy to `artifacts/models/<run>.keras`. |
| `EarlyStopping` | Stops when `val_loss` stops improving (patience 15, `min_delta 1e-4`). |
| `ReduceLROnPlateau` | Halves the learning rate after 5 stagnant epochs, down to `1e-6`. |
| `CSVLogger` | Writes a per-epoch log to `artifacts/logs/<run>.csv`. |
| `TerminateOnNaN` | Aborts immediately if the loss diverges to NaN. |

**Augmentation** (training split only — validation/test/inference are never augmented):

| Transform | Factor |
|---|---|
| `RandomFlip` | horizontal |
| `RandomTranslation` | ±0.125 (reflect fill) |
| `RandomRotation` | ±0.10 (reflect fill) |
| `RandomZoom` | ±0.10 (reflect fill) |

Augmentation is implemented with Keras preprocessing layers inside the training `tf.data` map, so it runs on the batch tensor and is automatically inactive at inference time.

---

## Project Structure

```
cnn-object-recognition/
├── cifar10_cnn/                 # the installable package
│   ├── __init__.py              # public re-exports (config surface)
│   ├── config.py                # paths, CLASS_NAMES, TrainConfig — single source of truth
│   ├── seeding.py               # global RNG seeding
│   ├── runtime.py               # UTF-8 console + GPU memory growth
│   ├── data.py                  # CIFAR-10 load, split, tf.data pipelines
│   ├── augmentation.py          # RandomFlip / Translation / Rotation / Zoom stack
│   ├── model.py                 # CNN architecture, compile, count_parameters
│   ├── callbacks.py             # checkpoint / early-stop / LR schedule / CSV / NaN
│   ├── metrics.py               # TF-free metrics (top-k, CE loss, confusion, summary)
│   ├── plots.py                 # matplotlib / seaborn report figures
│   ├── train.py                 # training entry point   → python -m cifar10_cnn.train
│   ├── evaluate.py              # evaluation entry point → python -m cifar10_cnn.evaluate
│   └── predict.py               # inference entry point  → python -m cifar10_cnn.predict
├── tests/                       # pytest suite (68 tests)
├── artifacts/                   # generated outputs (git-ignored)
│   ├── models/                  # <run_name>.keras — best checkpoint
│   ├── reports/                 # figures, JSON metrics, history, raw predictions
│   └── logs/                    # <run_name>.csv — per-epoch log
├── conftest.py                  # puts the project root on sys.path for pytest
├── requirements.txt             # pinned dependency set
├── CNN Object Detection.ipynb   # self-contained notebook version of the pipeline
└── README.md
```

---

## Installation

**Requirements:** Python 3.9–3.11 and TensorFlow 2.13–2.16 (developed and verified on **Python 3.11.5 / TensorFlow 2.15.0**).

### 1. Clone

```bash
git clone https://github.com/<your-username>/cifar10-cnn.git
cd cifar10-cnn
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or, with conda:

```bash
conda install -c conda-forge tensorflow numpy pandas scikit-learn matplotlib seaborn pytest
```

The package itself needs no installation step — it is imported from the repository root. (If you prefer, `pip install -e .` once a packaging file is added.)

> **Note on TensorFlow 2.15:** importing TensorFlow prints oneDNN and deprecation warnings. These are benign and expected; the test suite runs with one unrelated third-party Jupyter `DeprecationWarning`.

---

## Quick Start

Run a fast smoke test first — it exercises the entire pipeline in a couple of minutes on CPU (2 epochs over a 2,000-image subset) and confirms every artifact is produced before you commit to a long run:

```bash
python -m cifar10_cnn.train --quick
python -m cifar10_cnn.evaluate --quick
python -m cifar10_cnn.predict --index 7 --top-k 5
```

Then run the full job:

```bash
python -m cifar10_cnn.train --epochs 60 --batch-size 64
python -m cifar10_cnn.evaluate
```

---

## Usage

### Command line

Every stage is a runnable module.

#### Train

```bash
python -m cifar10_cnn.train [options]
```

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `60` | Number of epochs. |
| `--batch-size` | `64` | Batch size. |
| `--learning-rate` | `1e-3` | Adam learning rate. |
| `--seed` | `42` | Global random seed. |
| `--validation-split` | `0.1` | Fraction of the training set held out for validation. |
| `--run-name` | `cifar10_cnn` | Names every artifact produced by the run. |
| `--no-augmentation` | off | Disable all augmentation (baseline ablation). |
| `--quick` | off | Smoke test: 2 epochs on a 2,000-image subset. |

#### Evaluate

```bash
python -m cifar10_cnn.evaluate [options]
```

| Flag | Default | Description |
|---|---|---|
| `--run-name` | `cifar10_cnn` | Which run's checkpoint to evaluate. |
| `--batch-size` | `64` | Inference batch size. |
| `--model-path` | — | Explicit checkpoint path, overriding `--run-name`. |
| `--quick` | off | Evaluate on the 2,000-image smoke-test subset. |

Evaluate loads the best checkpoint, scores the untouched test set, prints a full report, and writes every figure and metric.

#### Predict

```bash
python -m cifar10_cnn.predict [options]
```

| Flag | Default | Description |
|---|---|---|
| `--image` | — | Path to an image file (JPG / JPEG / PNG / BMP / GIF / WEBP). |
| `--index` | — | CIFAR-10 test-set index to classify instead of a file. |
| `--top-k` | `3` | How many ranked predictions to show. |
| `--run-name` | `cifar10_cnn` | Which run's checkpoint to load. |
| `--model-path` | — | Explicit checkpoint path, overriding `--run-name`. |

Provide **exactly one** of `--image` or `--index`. An image file is resized to 32×32 automatically.

```bash
# Classify a sample straight out of the CIFAR-10 test set
python -m cifar10_cnn.predict --index 7 --top-k 5

# Classify any image on disk
python -m cifar10_cnn.predict --image path/to/your/image.jpg
```

### Python API

Each stage is a plain function you can call from your own code.

```python
from cifar10_cnn.config import TrainConfig
from cifar10_cnn.train import train
from cifar10_cnn.evaluate import run_evaluation
from cifar10_cnn.predict import run_prediction

config = TrainConfig(epochs=60, batch_size=64, run_name="my_run")

report = train(config)                 # writes artifacts/models/my_run.keras
evaluation = run_evaluation(config)    # metrics + figures under artifacts/reports/
result = run_prediction(config, image_path=None, index=7, top_k=5)
```

Build the model without training, for inspection or custom loops:

```python
from cifar10_cnn.model import build_model, count_parameters
from cifar10_cnn.config import TrainConfig

model = build_model(TrainConfig())
print(count_parameters(model))         # {'trainable': 814826, 'non_trainable': 1408, 'total': 816234}
```

### Notebook

`CNN Object Detection.ipynb` is a **self-contained** version of the entire pipeline — it does not import the `cifar10_cnn` package and can be run top to bottom in Jupyter or Anaconda. It is useful for interactive exploration.

```bash
jupyter notebook "CNN Object Detection.ipynb"
```

Set `CONFIG.quick_run = True` in section 7.1 for a smoke test before the full run.

---

## Output Artifacts

Everything a run produces lands under `artifacts/` (git-ignored), namespaced by `run_name`:

| Path | Contents |
|---|---|
| `models/<run_name>.keras` | Best checkpoint, selected by validation accuracy. |
| `reports/<run_name>_history.json` | Full per-epoch history plus a run summary. |
| `reports/<run_name>_config.json` | The exact serialised `TrainConfig` for the run. |
| `reports/<run_name>_metrics.json` | All evaluation numbers, including the confusion matrix. |
| `reports/<run_name>_predictions.npz` | Raw `labels`, `predictions` and `probabilities` arrays. |
| `reports/<run_name>_training_curves.png` | Loss and accuracy curves (train vs validation). |
| `reports/<run_name>_confusion_matrix.png` | Confusion matrix (counts). |
| `reports/<run_name>_confusion_matrix_normalized.png` | Row-normalised confusion matrix. |
| `reports/<run_name>_per_class_accuracy.png` | Recall per class. |
| `reports/<run_name>_sample_predictions.png` | Grid of labelled test predictions. |
| `logs/<run_name>.csv` | Per-epoch CSV log written by `CSVLogger`. |

Raw predictions are persisted to `.npz`, so the analysis can be revisited later without re-running the model.

---

## Configuration Reference

All tunables live on `cifar10_cnn/config.py::TrainConfig`, a dataclass with defaults tuned for CPU-only training.

| Field | Default | Description |
|---|---|---|
| `seed` | `42` | Seeds Python, NumPy and TensorFlow. |
| `batch_size` | `64` | Training batch size. |
| `epochs` | `60` | Maximum epochs (early stopping may cut this short). |
| `learning_rate` | `1e-3` | Initial Adam learning rate. |
| `validation_split` | `0.1` | Fraction held out from the training split. |
| `shuffle_buffer` | `10_000` | `tf.data` shuffle buffer size. |
| `conv_filters` | `(32, 64, 128)` | Filters per convolutional stage. |
| `conv_blocks_per_stage` | `2` | `Conv→BN→ReLU` blocks per stage. |
| `dense_units` | `256` | Units in the dense head. |
| `dropout_conv` | `(0.20, 0.30, 0.40)` | Dropout per convolutional stage. |
| `dropout_dense` | `0.50` | Dropout before the output layer. |
| `weight_decay` | `1e-4` | L2 regularisation strength. |
| `use_augmentation` | `True` | Master switch for augmentation. |
| `flip_horizontal` | `True` | Random horizontal flip. |
| `translation` | `0.125` | Random translation factor. |
| `rotation` | `0.10` | Random rotation factor. |
| `zoom` | `0.10` | Random zoom factor. |
| `early_stopping_patience` | `15` | Patience for `EarlyStopping`. |
| `reduce_lr_patience` | `5` | Patience for `ReduceLROnPlateau`. |
| `reduce_lr_factor` | `0.5` | LR multiplier on plateau. |
| `min_learning_rate` | `1e-6` | Floor for the LR schedule. |
| `quick_run` | `False` | 2 epochs on a 2,000-image subset. |
| `run_name` | `cifar10_cnn` | Namespace for every artifact. |
| `tags` | `{}` | Free-form metadata, serialised with the config. |

Changing `run_name` keeps multiple experiments side by side, since every artifact name derives from it.

---

## Testing

A pytest suite of **68 tests** covers every unit-testable module; the training/evaluation flows themselves are exercised end to end through the CLI.

```bash
python -m pytest -q
# 68 passed, 1 warning
```

Run a single module:

```bash
python -m pytest tests/test_metrics.py -q
```

| Test file | Module under test |
|---|---|
| `tests/test_config.py` | `config.py` — paths, defaults, serialisation |
| `tests/test_data.py` | `data.py` — normalisation, splitting, `tf.data` pipelines |
| `tests/test_model.py` | `model.py` — shapes, layers, parameter counts |
| `tests/test_metrics.py` | `metrics.py` — TF-free metric helpers |
| `tests/test_augmentation.py` | `augmentation.py` — layer stack and activation logic |
| `tests/test_callbacks.py` | `callbacks.py` — callback types and monitors |
| `tests/test_plots.py` | `plots.py` — writes real PNGs to a temp path |
| `tests/test_predict.py` | `predict.py` — ranking and input validation |
| `tests/test_seeding.py` | `seeding.py` — reproducibility |
| `tests/test_runtime.py` | `runtime.py` — device and console setup |

The single warning is a third-party Jupyter `DeprecationWarning`, unrelated to this project.

---

## Reproducibility

`set_global_seed(seed)` seeds `PYTHONHASHSEED`, Python's `random`, NumPy, and TensorFlow, and sets `TF_DETERMINISTIC_OPS=1`. The complete `TrainConfig` is written to `reports/<run_name>_config.json` for every run, so any result can be traced back to the exact settings that produced it.

Full bit-for-bit determinism on GPU additionally requires disabling cuDNN autotuning, which costs performance — seeding the standard generators is enough to make runs comparable across attempts.

---

## Results

> **No measured results are committed to this repository yet.** The `artifacts/` directory is git-ignored, and no trained checkpoint or numbers are published.

To record your own results, run the pipeline and paste the figures and numbers in:

```bash
python -m cifar10_cnn.train --epochs 60
python -m cifar10_cnn.evaluate
```

Then report the metrics printed by `evaluate` (top-3 accuracy, macro/weighted F1, per-class recall) and reference the generated figures:

```markdown
| Metric           | Value |
|------------------|-------|
| Test accuracy    | _your value_ |
| Top-3 accuracy   | _your value_ |
| Macro F1         | _your value_ |
| Weighted F1      | _your value_ |
| Epochs trained   | _your value_ |
| Training hardware| _your value_ |

![Training history](artifacts/reports/cifar10_cnn_training_curves.png)
![Confusion matrix](artifacts/reports/cifar10_cnn_confusion_matrix.png)
```

For context, a from-scratch CNN of this depth typically lands in the **low-to-mid 80s percent** on CIFAR-10 with augmentation — but do not publish a figure you have not measured yourself.

---

## Design Notes

- **Framing.** The input pipeline is split so that pure logic (`normalize_images`, `split_train_validation`) is unit-testable without network access or a GPU. Augmentation is a first-class module, not inlined into the training loop.
- **Metrics without TensorFlow.** `metrics.py` deliberately avoids importing TensorFlow so evaluation logic can be tested against plain NumPy arrays. This matters because evaluation loads the model with `compile=False` (only the forward pass is needed, and it avoids failures on missing optimizers or legacy configs).
- **Two monitors.** Checkpointing uses validation *accuracy* (the metric we care about); early stopping and the LR schedule use validation *loss* (a smoother signal, so those decisions are less noisy).
- **Memory.** Images are cached as `uint8` (~150 MB for the full training set) and only converted to `float32` per batch, which keeps the dataset in RAM without the 4× blow-up of caching float data.
- **Runtime setup.** `runtime.py` forces the console to UTF-8 so Keras' box-drawing characters in `model.summary()` never raise `UnicodeEncodeError` on Windows, and enables incremental GPU memory growth when a GPU is present.

---

## Roadmap

Ideas for extending the project, roughly in order of value:

- [ ] Publish measured results and commit the confusion-matrix figure.
- [ ] Add an end-to-end smoke test (opt-in marker) that runs `--quick` and asserts the artifacts appear.
- [ ] Add CLI argument-parsing tests (`train.parse_args`, `evaluate.parse_args`).
- [ ] Add `pyproject.toml` / `pytest.ini` to centralise tooling config and register `slow` / `network` markers.
- [ ] Explore deeper or residual architectures, mixup/cutmix augmentation, and cosine LR schedules.
- [ ] Add a `--no-augmentation` vs. augmented comparison table to the Results section.

---

## License

No license file is currently included. Add one (e.g. the MIT License) before distributing, or specify the terms explicitly.
