"""
Fashion MNIST Classification with a 6-Layer Convolutional Neural Network
========================================================================
Author : Junior ML Researcher, Microsoft AI
Purpose: Classify the Fashion MNIST dataset using a CNN built with Keras,
         implemented with Python classes (OOP best practices). The same
         architecture can later be adapted for user-profile image
         classification in marketing-targeting projects.

The network contains SIX trainable/structural layers:
    1. Conv2D        (32 filters, 3x3, ReLU)
    2. MaxPooling2D  (2x2)
    3. Conv2D        (64 filters, 3x3, ReLU)
    4. MaxPooling2D  (2x2)
    5. Dense         (128 units, ReLU)   -- after a Flatten reshape
    6. Dense         (10 units, Softmax) -- output layer

Usage:
    python fashion_mnist_cnn.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Human-readable class names for the 10 Fashion MNIST labels (0-9)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionMnistData:
    """Loads and pre-processes the Fashion MNIST dataset."""

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load(self):
        """Load the dataset and normalise pixel values to [0, 1]."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        # Scale to [0, 1] and add a channel dimension: (28, 28) -> (28, 28, 1)
        self.x_train = x_train.astype("float32") / 255.0
        self.x_test = x_test.astype("float32") / 255.0
        self.x_train = np.expand_dims(self.x_train, -1)
        self.x_test = np.expand_dims(self.x_test, -1)

        self.y_train = y_train
        self.y_test = y_test

        print(f"Training samples: {self.x_train.shape[0]}")
        print(f"Test samples    : {self.x_test.shape[0]}")
        return self


class FashionMnistCNN:
    """A six-layer CNN classifier for Fashion MNIST."""

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """Construct the six-layer CNN."""
        model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                # Layer 1: convolution
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                # Layer 2: pooling
                layers.MaxPooling2D(pool_size=(2, 2)),
                # Layer 3: convolution
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                # Layer 4: pooling
                layers.MaxPooling2D(pool_size=(2, 2)),
                # Reshape only (not counted as a layer)
                layers.Flatten(),
                # Layer 5: fully connected
                layers.Dense(128, activation="relu"),
                # Layer 6: output
                layers.Dense(self.num_classes, activation="softmax"),
            ],
            name="fashion_mnist_cnn",
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def summary(self):
        self.model.summary()

    def train(self, x_train, y_train, epochs=10, batch_size=128, validation_split=0.1):
        """Train the network and return the training history."""
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=2,
        )
        return history

    def evaluate(self, x_test, y_test):
        """Evaluate on the held-out test set."""
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest loss    : {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        return loss, accuracy

    def predict_images(self, images, true_labels=None, save_plot="predictions.png"):
        """
        Make predictions for one or more images and (optionally) save a
        visualisation comparing predictions to the true labels.
        """
        probabilities = self.model.predict(images, verbose=0)
        predicted_labels = np.argmax(probabilities, axis=1)

        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.imshow(images[i].squeeze(), cmap="gray")
            pred_name = CLASS_NAMES[predicted_labels[i]]
            conf = probabilities[i][predicted_labels[i]] * 100
            title = f"Predicted: {pred_name} ({conf:.1f}%)"
            if true_labels is not None:
                title += f"\nActual: {CLASS_NAMES[true_labels[i]]}"
            ax.set_title(title)
            ax.axis("off")

            print(
                f"Image {i + 1}: predicted = {pred_name} "
                f"({conf:.2f}% confidence)"
                + (f", actual = {CLASS_NAMES[true_labels[i]]}" if true_labels is not None else "")
            )

        plt.tight_layout()
        plt.savefig(save_plot, dpi=150)
        print(f"Prediction visualisation saved to '{save_plot}'")
        return predicted_labels, probabilities

    def save(self, path="fashion_mnist_cnn.keras"):
        self.model.save(path)
        print(f"Model saved to '{path}'")


def main():
    # 1. Load and pre-process the data
    data = FashionMnistData().load()

    # 2. Build the six-layer CNN
    cnn = FashionMnistCNN()
    cnn.summary()

    # 3. Train
    cnn.train(data.x_train, data.y_train, epochs=10)

    # 4. Evaluate
    cnn.evaluate(data.x_test, data.y_test)

    # 5. Predict at least two images from the test set
    sample_images = data.x_test[:2]
    sample_labels = data.y_test[:2]
    cnn.predict_images(sample_images, true_labels=sample_labels)

    # 6. Persist the trained model for later reuse / adaptation
    cnn.save()


if __name__ == "__main__":
    main()
