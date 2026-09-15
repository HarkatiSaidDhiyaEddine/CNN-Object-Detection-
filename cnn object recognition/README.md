# CIFAR-10 Object Recognition with Convolutional Neural Networks

A convolutional neural network (CNN) built with TensorFlow/Keras that classifies 32×32 color images into 10 object categories. The project covers the full workflow: data loading and normalization, CNN architecture design, training, evaluation, prediction, and confusion-matrix analysis.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Requirements](#requirements)
- [Notes and Improvements](#notes-and-improvements)
- [License](#license)

---

## Overview

This project implements an image classification pipeline for the CIFAR-10 dataset using a custom CNN. The model is trained from scratch (no transfer learning) and produces a class prediction for each input image, followed by a detailed error analysis via confusion matrix.

**Pipeline stages:**

1. Load CIFAR-10 through `tensorflow.keras.datasets`.
2. Normalize pixel values to the `[0, 1]` range.
3. Build a sequential CNN with four convolution layers, two max-pooling layers, dropout, and two dense layers.
4. Compile with the Adam optimizer and sparse categorical cross-entropy loss.
5. Train for 30 epochs.
6. Evaluate on the held-out test set.
7. Predict labels, compute a confusion matrix, and visualize it as a heatmap.

---

## Dataset

**CIFAR-10** — 60,000 labeled 32×32 RGB images across 10 mutually exclusive classes:

| Index | Class      |
|-------|------------|
| 0     | Airplane   |
| 1     | Automobile |
| 2     | Bird       |
| 3     | Cat        |
| 4     | Deer       |
| 5     | Dog        |
| 6     | Frog       |
| 7     | Horse      |
| 8     | Ship       |
| 9     | Truck      |

- **Training set:** 50,000 images
- **Test set:** 10,000 images
- **Image shape:** `(32, 32, 3)`
- **Normalization:** pixel values divided by `255.0`

The dataset is downloaded automatically by Keras on first run:

```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

---

## Model Architecture

A `tf.keras.models.Sequential` model with the following layer stack:

| # | Layer              | Configuration                                      |
|---|--------------------|----------------------------------------------------|
| 1 | Conv2D             | 32 filters, 3×3 kernel, `same` padding, ReLU, input shape `(32, 32, 3)` |
| 2 | Conv2D             | 32 filters, 3×3 kernel, `same` padding, ReLU       |
| 3 | MaxPool2D          | pool size 2, stride 2, `valid` padding             |
| 4 | Conv2D             | 64 filters, 3×3 kernel, `same` padding, ReLU       |
| 5 | Conv2D             | 64 filters, 3×3 kernel, `same` padding, ReLU       |
| 6 | MaxPool2D          | pool size 2, stride 2, `valid` padding             |
| 7 | Dropout            | rate 0.1                                           |
| 8 | Flatten            | —                                                  |
| 9 | Dense              | 128 units, ReLU                                    |
| 10 | Dense (output)    | 10 units, Softmax                                  |

**Design rationale:**

- **Two conv blocks** (conv → conv → pool) allow the network to learn increasingly abstract features at two spatial scales before downsampling.
- **Small 3×3 kernels** with `same` padding preserve spatial dimensions within a block and keep the parameter count low.
- **Filter count 32 → 64** doubles channel depth as spatial resolution halves, keeping an information bottleneck that grows in abstraction.
- **Dropout (0.1)** provides light regularization against overfitting.
- **Softmax output** yields a probability distribution over the 10 classes.

---

## Training Configuration

| Parameter        | Value                             |
|------------------|-----------------------------------|
| Loss function    | `sparse_categorical_crossentropy` |
| Optimizer        | Adam (default learning rate)      |
| Metric           | `sparse_categorical_accuracy`     |
| Batch size       | 32                                |
| Epochs           | 30                                |
| Input size       | 32×32×3                           |
| Output classes   | 10                                |

Sparse categorical cross-entropy is used because the labels are provided as integer class indices rather than one-hot vectors.

---

## Evaluation

Performance is measured on the 10,000-image test set:

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)
```

Predictions are obtained by taking the argmax over the softmax output:

```python
y_pred = np.argmax(model.predict(x_test), axis=-1)
```

### Confusion Matrix

A confusion matrix is computed with scikit-learn and visualized with a seaborn heatmap, showing how often each true class is predicted as each class — which reveals which object categories the model confuses most often (typically visually similar pairs such as cat/dog and automobile/truck).

```python
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
```

---

## Results

> **Placeholder — no measured results are committed to this repository yet.**
>
> The notebook does not persist its training output, and no saved metrics or model weights are included. Record the real numbers from your own run before publishing:
>
> | Metric             | Value |
> |--------------------|-------|
> | Test loss          | _TBD_ |
> | Test accuracy      | _TBD_ |
> | Epochs trained     | 30    |
> | Training hardware  | _TBD_ |
>
> Then save the heatmap produced by the final cell to `images/confusion_matrix.png` and reference it here:
>
> ```markdown
> ![Confusion Matrix](images/confusion_matrix.png)
> ```
>
> For context, a from-scratch CNN of this depth on unaugmented CIFAR-10 typically lands in the low-to-mid 70s percent range — but do not publish a figure you have not measured yourself.

---

## Project Structure

```
cnn-object-recognition/
└── Untitled.ipynb   # Full pipeline: preprocessing, model, training, evaluation
```

Everything — data loading, normalization, model definition, compilation, training, evaluation, prediction, and visualization — lives in a single Jupyter notebook.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open `Untitled.ipynb` and run the cells in order.

---

## Usage

The notebook is structured to be executed top-to-bottom:

1. **Imports** — TensorFlow, NumPy, pandas, matplotlib.
2. **Data loading and preprocessing** — CIFAR-10 download, normalization to `[0, 1]`, sample image visualization.
3. **Model definition** — sequential CNN built layer by layer.
4. **Training** — 30 epochs at batch size 32.
5. **Evaluation** — test loss and accuracy.
6. **Prediction** — per-image predictions compared against true labels.
7. **Confusion matrix** — numeric matrix plus seaborn heatmap.

### Inference on a single image

```python
import numpy as np

# image: a (32, 32, 3) array with pixel values in [0, 1]
image = np.expand_dims(image, axis=0)          # shape (1, 32, 32, 3)
predicted_index = np.argmax(model.predict(image), axis=-1)[0]

print(class_names[predicted_index])
```

### Saving and reloading the model

```python
model.save("cifar10_cnn.keras")                # save
model = tf.keras.models.load_model("cifar10_cnn.keras")   # reload
```

**Note:** the notebook as written does not persist the trained model, so training is repeated on every run. Adding the save call above is recommended.

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook / JupyterLab

Training on CPU is feasible but slow; a CUDA-capable GPU dramatically reduces training time.

---

## Notes and Improvements

The current model is a solid baseline. Possible enhancements:

- **Data augmentation** (`RandomFlip`, `RandomRotation`, `RandomZoom`) to reduce overfitting and improve generalization.
- **Batch Normalization** after convolution layers for faster, more stable convergence.
- **Learning-rate scheduling** or a lower learning rate with more epochs.
- **Higher dropout** rates, or dropout placed after each conv block.
- **Additional conv blocks** or residual connections to increase capacity.
- **Model checkpointing** (`ModelCheckpoint`) to save the best epoch rather than the last.
- **Early stopping** to avoid unnecessary training once validation loss plateaus.
- **A held-out validation split** during training to monitor generalization per epoch.

---

## License

No license file is currently included. Add one (e.g. MIT) before distributing, or specify the terms explicitly.
