## Simple Neural Network for MNIST Digit Classification

This project implements a basic deep neural network (DNN) from scratch using only NumPy to classify handwritten digits in the MNIST dataset.
It includes data loading, preprocessing, model training, hyperparameter tuning, and evaluation.

---

### Project Structure

```
.
├── dataset
│   ├── t10k-images.idx3-ubyte          # MNIST test images (raw format)
│   ├── t10k-labels.idx1-ubyte          # MNIST test labels (raw format)
│   ├── train-images.idx3-ubyte         # MNIST training images (raw format)
│   ├── train-labels.idx1-ubyte         # MNIST training labels (raw format)
│   ├── test_images.npy                 # Preprocessed test images
│   └── test_labels.npy                 # Preprocessed test labels
├── dnn.py                              # Main script for running and training the network
├── mnistdatareader.py                  # Module to load and preprocess MNIST data
├── neuralnetwork.py                    # Neural network and layer definitions
├── best_hyperparams.pkl                # Saved optimal hyperparameters
├── hyperparameter_tuning_results.pkl   # Tuning results for analysis
└── training_state.pkl                  # Model state for training resumption (weights, biases, epochs)
```

### Neural Network Architecture

**Input Layer**: 784 nodes (28x28 pixels)

**Hidden Layer**: 512 neurons with ReLU activation

**Output Layer**: 10 neurons with softmax activation (digits 0-9)

### Data

The project uses the MNIST dataset of grayscale images:
- **Training Set**: 60,000 images with labels
- **Test Set**: 10,000 images with labels
Each image is a 28x28 pixel grid flattened into a 1D vector of 784 values.

### Key Features
**Data Handling**: Loads and preprocesses MNIST data. The data is saved in `.npy` format for faster loading.

**Training and Evaluation**: Includes functions to train and validate the model, with option to resume training from a saved state.

**Hyperparameter Tuning**: Searches for optimal learning rate, batch size, and decay factor. Saves best hyperparameters for future runs.

### How to use?

1. **Data Preparation**: Run `dnn.py` to load the data from MNIST `.idx` files in the `dataset` folder, preprocess, and save `.npy` files.
2. **Hyperparameter Tuning**: Once the data is loaded, `dnn.py` automatically tunes and saves best hyperparameters in `best_hyperparams.pkl`.
3. **Training**: Once, the best hyperparameters are found, the training will start.
4. **Testing**: After training, the model's performance is evaluated on the test set, with a final loss displayed.

### Results

The final model can classify images from the test set.

### Requirements

- `numpy`
- `matplotlib`

---

