import numpy as np
import matplotlib.pyplot as plt
import os
from mnistdatareader import load_labels, load_images, load_or_save_data
from neuralnetwork import NeuralNetwork
import itertools
import pickle


# reshape each image to a 2D 28x28 format
def reshape_matrix_to_images(images):
    return images.reshape(-1, 28, 28)


# display a few images and their labels
def display_samples(images, labels, num_samples=3):
    for i in range(num_samples):
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
        plt.show()

# display some predictions
def display_predictions(nn, images, labels, num_samples=5):
    predictions = nn.forward(images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    for i in range(num_samples):
        plt.imshow(images[i].reshape(28,28), cmap='gray')
        plt.title(f"True Label: {labels[i]} - Predicted: {predicted_labels[i]}")
        plt.axis('off')
        plt.show()

# save best hyperparameters to a file
def save_best_params(best_params, filename="best_hyperparams.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(best_params, f)
    print(f"Best hyperparameters saved to {filename}")

# load best hyperparameters from a file if it exists
def load_best_params(filename="best_hyperparams.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            best_params = pickle.load(f)
        print(f"Best hyperparameters loaded from {filename}")
        print("Best Hyperparameters:", best_params)
        return best_params
    else:
        print("No saved best hyperparameters found. Running hyperparameter tuning...")
        return None

def hyperparameter_tuning(X, y, epochs=5, trials=5, sample_size=3000, learning_rates=[0.01, 0.001], batch_sizes=[32, 64], decays=[0.99, 0.95]):
    """
    Function to find the best hyperparameters for the neural network.
    
    Parameters:
    - nn: The NeuralNetwork object.
    - X, y: Training data and labels.
    - epochs: Number of epochs to train for each trial.
    - trials: Number of different trials to run.
    - learning_rates: List of learning rates to try.
    - batch_sizes: List of batch sizes to try.
    - decays: List of decay rates for the learning rate to try.

    Returns:
    - best_params: Dictionary of the best hyperparameters found.
    - best_val_loss: Lowest validation loss achieved.
    """
    best_val_loss = float('inf')
    best_params = None
    results = []

    # shuffle training data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]


    for lr, batch_size, decay in itertools.product(learning_rates, batch_sizes, decays,):
        print(f"Testing combination: Learning rate={lr}, Batch size={batch_size}, Decay={decay}")
        
        trial_val_losses = []
        
        for trial in range(trials):
            nn = NeuralNetwork(learning_rate=lr)

            # training the model on a subset of data (e.g., 3000 samples)
            subset_size = sample_size       
            X_subset, y_subset = X[:subset_size], y[:subset_size]
            
            train_losses, val_losses = nn.train(X_subset, y_subset, epochs=epochs, batch_size=batch_size, decay=decay)
            
            final_val_loss = val_losses[-1]
            trial_val_losses.append(final_val_loss)
            print(f"Trial {trial + 1}/{trials} - Validation loss: {final_val_loss:.4f}")

        avg_val_loss = np.mean(trial_val_losses)
        results.append((lr, batch_size, decay, avg_val_loss))
        print(f"Average validation loss for this combination: {avg_val_loss:.4f}")

        # update best hyperparameters if the average loss is the lowest
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_params = {
                "learning_rate": lr,
                "batch_size": batch_size,
                "decay": decay,
            }

    if best_params:
        save_best_params(best_params)

    # save results to a file for later analysis
    with open("hyperparameter_tuning_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Hyperparameter tuning completed. Results saved to hyperparameter_tuning_results.pkl")

    return best_params, best_val_loss


dataset_path = './dataset/'
# load training and testing data, checking for .npy files first
train_images = load_or_save_data(load_images, os.path.join(dataset_path, 'train-images.idx3-ubyte'), 'train_images', dataset_path)
train_labels = load_or_save_data(load_labels, os.path.join(dataset_path, 'train-labels.idx1-ubyte'), 'train_labels', dataset_path)
test_images = load_or_save_data(load_images, os.path.join(dataset_path, 't10k-images.idx3-ubyte'), 'test_images', dataset_path)
test_labels = load_or_save_data(load_labels, os.path.join(dataset_path, 't10k-labels.idx1-ubyte'), 'test_labels', dataset_path)

# normalization
train_images_mean = np.mean(train_images)
train_images_std = np.std(train_images)

train_images_normalized = (train_images - train_images_mean) / train_images_std
test_images_normalized = (test_images - train_images_mean) / train_images_std


train_images_reshaped = reshape_matrix_to_images(train_images_normalized)
test_images_reshaped = reshape_matrix_to_images(test_images_normalized)
# display_samples(train_images_reshaped, train_labels)


# prepare data for training (reshape to 1D vectors)
train_images_flattened = train_images_normalized.reshape(-1, 784)
test_images_flattened = test_images_normalized.reshape(-1, 784)

# check for existing best hyperparameters or run tuning
best_params = load_best_params()
if not best_params:
    best_params, best_val_loss = hyperparameter_tuning(
        train_images_flattened, train_labels, 
        epochs=5, trials=3, sample_size=3000,
        learning_rates=[0.01, 0.001, 0.02],
        batch_sizes=[32, 64, 20, 75],
        decays=[0.99, 0.95, 0.92],
    )
    print("Best Hyperparameters:")
    print(best_params)
    print(f"Validation Loss for those params: {best_val_loss:.4f}")

# train the neural network
print("Starting training...")
nn = NeuralNetwork(learning_rate=best_params['learning_rate'])
train_losses, val_losses = nn.train(train_images_flattened, train_labels, epochs=10, batch_size=best_params['batch_size'], decay=best_params['decay'], resume=False)

# evaluate the model on test data
print("Evaluating on test set...")
y_test_pred = nn.forward(test_images_flattened)
test_loss = nn.calculate_loss(y_test_pred, test_labels)
print(f"Test Loss: {test_loss:.4f}")


# display some predictions
# display_predictions(nn, test_images_flattened, test_labels)