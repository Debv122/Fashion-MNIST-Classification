# =====================================================================
# Fashion MNIST Classification with a 6-Layer Convolutional Neural Network
# =====================================================================
# Author : Junior ML Researcher, Microsoft AI
# Purpose: Classify the Fashion MNIST dataset using a CNN built with the
#          Keras R interface, implemented with R Reference Classes (R5),
#          mirroring the Python implementation in fashion_mnist_cnn.py.
#
# The network contains SIX trainable/structural layers:
#   1. Conv2D       (32 filters, 3x3, ReLU)
#   2. MaxPooling2D (2x2)
#   3. Conv2D       (64 filters, 3x3, ReLU)
#   4. MaxPooling2D (2x2)
#   5. Dense        (128 units, ReLU)   -- after a Flatten reshape
#   6. Dense        (10 units, Softmax) -- output layer
#
# Requirements:
#   install.packages("keras3")   # or install.packages("keras")
#   keras3::install_keras()      # installs the TensorFlow backend
#
# Usage:
#   Rscript fashion_mnist_cnn.R
# =====================================================================

library(keras3)   # use library(keras) if you have the older package

CLASS_NAMES <- c(
  "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
)

# ---------------------------------------------------------------------
# Data class: loads and pre-processes Fashion MNIST
# ---------------------------------------------------------------------
FashionMnistData <- setRefClass(
  "FashionMnistData",
  fields = list(
    x_train = "ANY",
    y_train = "ANY",
    x_test  = "ANY",
    y_test  = "ANY"
  ),
  methods = list(
    load = function() {
      "Load the dataset and normalise pixel values to [0, 1]."
      fashion <- dataset_fashion_mnist()

      x_train <<- fashion$train$x / 255
      x_test  <<- fashion$test$x  / 255

      # Add a channel dimension: (n, 28, 28) -> (n, 28, 28, 1)
      dim(x_train) <<- c(dim(x_train), 1)
      dim(x_test)  <<- c(dim(x_test), 1)

      y_train <<- fashion$train$y
      y_test  <<- fashion$test$y

      cat("Training samples:", dim(x_train)[1], "\n")
      cat("Test samples    :", dim(x_test)[1], "\n")
      invisible(.self)
    }
  )
)

# ---------------------------------------------------------------------
# Model class: the six-layer CNN
# ---------------------------------------------------------------------
FashionMnistCNN <- setRefClass(
  "FashionMnistCNN",
  fields = list(
    model = "ANY"
  ),
  methods = list(
    initialize = function(...) {
      callSuper(...)
      build_model()
    },

    build_model = function() {
      "Construct the six-layer CNN."
      model <<- keras_model_sequential(input_shape = c(28, 28, 1)) |>
        # Layer 1: convolution
        layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") |>
        # Layer 2: pooling
        layer_max_pooling_2d(pool_size = c(2, 2)) |>
        # Layer 3: convolution
        layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") |>
        # Layer 4: pooling
        layer_max_pooling_2d(pool_size = c(2, 2)) |>
        # Reshape only (not counted as a layer)
        layer_flatten() |>
        # Layer 5: fully connected
        layer_dense(units = 128, activation = "relu") |>
        # Layer 6: output
        layer_dense(units = 10, activation = "softmax")

      model |> compile(
        optimizer = "adam",
        loss      = "sparse_categorical_crossentropy",
        metrics   = "accuracy"
      )
      invisible(.self)
    },

    show_summary = function() {
      summary(model)
    },

    train = function(x_train, y_train, epochs = 10, batch_size = 128,
                     validation_split = 0.1) {
      "Train the network and return the training history."
      history <- model |> fit(
        x_train, y_train,
        epochs           = epochs,
        batch_size       = batch_size,
        validation_split = validation_split,
        verbose          = 2
      )
      invisible(history)
    },

    evaluate_model = function(x_test, y_test) {
      "Evaluate on the held-out test set."
      scores <- model |> evaluate(x_test, y_test, verbose = 0)
      cat(sprintf("\nTest loss    : %.4f\n", scores$loss))
      cat(sprintf("Test accuracy: %.4f\n", scores$accuracy))
      invisible(scores)
    },

    predict_images = function(images, true_labels = NULL,
                              save_plot = "predictions_r.png") {
      "Predict one or more images and save a visualisation."
      probabilities    <- model |> predict(images, verbose = 0)
      predicted_labels <- apply(probabilities, 1, which.max) - 1  # 0-indexed

      n <- dim(images)[1]
      png(save_plot, width = 400 * n, height = 420)
      par(mfrow = c(1, n), mar = c(1, 1, 4, 1))

      for (i in seq_len(n)) {
        img       <- images[i, , , 1]
        pred_name <- CLASS_NAMES[predicted_labels[i] + 1]
        conf      <- probabilities[i, predicted_labels[i] + 1] * 100

        title_txt <- sprintf("Predicted: %s (%.1f%%)", pred_name, conf)
        if (!is.null(true_labels)) {
          title_txt <- paste0(
            title_txt, "\nActual: ", CLASS_NAMES[true_labels[i] + 1]
          )
        }

        # Rotate the matrix so the image displays upright
        image(t(apply(img, 2, rev)), col = gray.colors(256, 0, 1),
              axes = FALSE, main = title_txt)

        cat(sprintf(
          "Image %d: predicted = %s (%.2f%% confidence)%s\n",
          i, pred_name, conf,
          if (!is.null(true_labels))
            paste0(", actual = ", CLASS_NAMES[true_labels[i] + 1]) else ""
        ))
      }
      dev.off()
      cat("Prediction visualisation saved to '", save_plot, "'\n", sep = "")
      invisible(list(labels = predicted_labels, probabilities = probabilities))
    },

    save_model = function(path = "fashion_mnist_cnn_r.keras") {
      save_model(model, path)
      cat("Model saved to '", path, "'\n", sep = "")
    }
  )
)

# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
main <- function() {
  # 1. Load and pre-process the data
  data <- FashionMnistData$new()
  data$load()

  # 2. Build the six-layer CNN
  cnn <- FashionMnistCNN$new()
  cnn$show_summary()

  # 3. Train
  cnn$train(data$x_train, data$y_train, epochs = 10)

  # 4. Evaluate
  cnn$evaluate_model(data$x_test, data$y_test)

  # 5. Predict at least two images from the test set
  sample_images <- data$x_test[1:2, , , , drop = FALSE]
  sample_labels <- data$y_test[1:2]
  cnn$predict_images(sample_images, true_labels = sample_labels)
}

main()
