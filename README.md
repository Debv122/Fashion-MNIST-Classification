# Fashion MNIST Classification — 6-Layer CNN (Python & R)

A Convolutional Neural Network built with Keras to classify the
[Fashion MNIST dataset](https://keras.io/api/datasets/fashion_mnist/),
implemented twice — once in **Python** (using Python classes) and once in
**R** (using R Reference Classes) — as preparation for adapting the same
architecture to user-profile image classification for targeted marketing.

## Contents

| File                   | Description                                                |
|------------------------|------------------------------------------------------------|
| `fashion_mnist_cnn.py` | Python implementation (OOP, two classes)                   |
| `fashion_mnist_cnn.R`  | R implementation (Reference Classes / R5)                  |
| `README.md`            | This file                                                  |

## The Dataset

Fashion MNIST contains 60,000 training and 10,000 test images. Each image
is a 28×28 grayscale picture belonging to one of 10 fashion categories:

| Label | Class       | Label | Class      |
|-------|-------------|-------|------------|
| 0     | T-shirt/top | 5     | Sandal     |
| 1     | Trouser     | 6     | Shirt      |
| 2     | Pullover    | 7     | Sneaker    |
| 3     | Dress       | 8     | Bag        |
| 4     | Coat        | 9     | Ankle boot |

The dataset is downloaded automatically by Keras on first run — no manual
download is needed.

## Network Architecture (6 Layers)

| # | Layer        | Configuration                  |
|---|--------------|--------------------------------|
| 1 | Conv2D       | 32 filters, 3×3 kernel, ReLU   |
| 2 | MaxPooling2D | 2×2 pool                       |
| 3 | Conv2D       | 64 filters, 3×3 kernel, ReLU   |
| 4 | MaxPooling2D | 2×2 pool                       |
| 5 | Dense        | 128 units, ReLU                |
| 6 | Dense        | 10 units, Softmax (output)     |

A `Flatten` operation sits between layers 4 and 5; it only reshapes the
tensor and has no trainable parameters, so it is not counted as one of the
six layers.

- **Optimizer:** Adam
- **Loss:** Sparse categorical cross-entropy
- **Training:** 10 epochs, batch size 128, 10% validation split
- **Expected test accuracy:** ~90–91%

## How to Run

### Python

Requirements: Python 3.9+, TensorFlow 2.x, NumPy, Matplotlib.

```bash
pip install tensorflow numpy matplotlib
python fashion_mnist_cnn.py
```

The script will:
1. Download and pre-process the dataset (pixels scaled to [0, 1], channel
   dimension added).
2. Build and print a summary of the six-layer CNN.
3. Train for 10 epochs and report validation accuracy per epoch.
4. Evaluate on the 10,000-image test set.
5. **Predict two test images**, print each prediction with its confidence,
   and save a side-by-side visualisation to `predictions.png`.
6. Save the trained model to `fashion_mnist_cnn.keras`.

### R

Requirements: R 4.x with the `keras3` (or `keras`) package and a
TensorFlow backend.

```r
install.packages("keras3")
keras3::install_keras()
```

Then run:

```bash
Rscript fashion_mnist_cnn.R
```

The R script performs the same workflow: load → build → train → evaluate →
**predict two test images** (visualisation saved to `predictions_r.png`).

## Code Structure

Both implementations follow the same object-oriented design:

- **`FashionMnistData`** — responsible for loading the dataset and
  pre-processing it (normalisation and reshaping).
- **`FashionMnistCNN`** — encapsulates the model: building, training,
  evaluation, prediction with visualisation, and saving.
- **`main()`** — orchestrates the full workflow end to end.

This separation of concerns makes it straightforward to later swap
`FashionMnistData` for a class that loads user-profile images, while
reusing or fine-tuning `FashionMnistCNN` for the marketing-targeting
project.

## Interpreting the Output

- Per-epoch logs show training and validation loss/accuracy — validation
  accuracy should climb to roughly 0.90.
- The final evaluation block reports test loss and test accuracy.
- The prediction step prints lines such as
  `Image 1: predicted = Ankle boot (99.12% confidence), actual = Ankle boot`
  and saves an image file showing each input alongside its predicted and
  actual class.

## Notes & Best Practices Applied

- Pixel normalisation to [0, 1] for stable training.
- `sparse_categorical_crossentropy` to avoid one-hot encoding labels.
- A validation split to monitor overfitting during training.
- Docstrings/comments, PEP 8-style Python, and idiomatic R throughout.
- Identical architectures in both languages so results are comparable.
