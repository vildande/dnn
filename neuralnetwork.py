# neural_network.py
import numpy as np
import pickle

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation=None):
        self.weights =  np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs) # xavier init
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation.lower() if activation else None
        self.output = None # output of the layer
        self.input = None # input from previous layer (for backprop)
        self.dweights = None # gradient of the loss with respect to the weights
        self.dbiases = None # gradient of the loss with respeect to the biases


    def forward(self, inputs):
        self.input = inputs # save input for backpropagation
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activation == "relu":
            self.output = np.maximum(0, self.output)
        elif self.activation == "softmax":
            exp_values = np.exp(self.output - np.max(self.output, axis=1, keepdims=True))
            self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def backward(self, dvalues):
        # (dvalues - the gradient of the loss with respect to the output layer, comes from layer ahead)

        # calculate gradients of the loss for weights and biases
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # calculate the gradient for the input for the current layer by "reversing" the influence of weights
        return np.dot(dvalues, self.weights.T)


class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.layers = [
            DenseLayer(784, 512, activation="relu"),
            DenseLayer(512, 10, activation="softmax")
        ]
        self.learning_rate = learning_rate

    def forward(self, X):
        out = X
        for layer in self.layers:
            layer.forward(out)
            out = layer.output
        return out

    def backward(self, y_true):
        sample_count = len(y_true)
        y_pred = self.layers[-1].output

        # if the targets are in form of scalar class values, convert them to one-hot
        if len(y_true.shape) == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        # loss gradient of the output of the nn
        dvalues = (y_pred - y_true) / sample_count

        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def update_params(self):
        for layer in self.layers:
            # update weights and biases
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases

    def calculate_loss(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # avoid log(0) and log(1) error
        
        if len(y_true.shape) == 1:  # if scalar class values
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  # if one-hot encoded vectors
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods) # average loss over batch
    
    def train(self, X, y, epochs=100, batch_size=32, decay=0.99, resume=False):
        start_epoch, train_losses, val_losses = 0, [], []  # default values if not resuming

        # load state if resuming training
        if resume:
            start_epoch, train_losses, val_losses = self.load_state()

        initial_learning_rate = self.learning_rate
        
        # loop through each epoch, resuming from the saved epoch if applicable
        for epoch in range(start_epoch, epochs):
            # adjust learning rate with decay factor
            self.learning_rate = initial_learning_rate * (decay ** epoch)

            # shuffle training data at the start of each epoch
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            # split into 75% training and 25% validation
            split_index = int(0.75 * X.shape[0])
            X_train, y_train = X[:split_index], y[:split_index]
            X_val, y_val = X[split_index:], y[split_index:]
            
            # process batches
            batch_losses = []
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                X_batch, y_batch = X_train[start:end], y_train[start:end]

                # Forward pass through the network
                y_pred = self.forward(X_batch)

                # Calculate loss and store it for this batch
                loss = self.calculate_loss(y_pred, y_batch)
                batch_losses.append(loss)

                # Backward pass and update parameters
                self.backward(y_batch)
                self.update_params()

            # calculate average training loss for the epoch
            avg_train_loss = np.mean(batch_losses)
            train_losses.append(avg_train_loss)

            
            # validation pass: calculate validation loss
            y_val_pred = self.forward(X_val)
            val_loss = self.calculate_loss(y_val_pred, y_val)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            
            self.save_state(epoch, train_losses, val_losses)


        return train_losses, val_losses

    def save_state(self, epoch, train_losses, val_losses):
        state = {
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "learning_rate": self.learning_rate,
            "layer_params": [
                {"weights": layer.weights, "biases": layer.biases}
                for layer in self.layers
            ]
        }
        with open("training_state.pkl", "wb") as f:
            pickle.dump(state, f)
        print(f"Training state saved at epoch {epoch + 1}")

    def load_state(self):
        try:
            with open("training_state.pkl", "rb") as f:
                state = pickle.load(f)
            
            # restore epoch, losses, and learning rate
            start_epoch = state["epoch"]
            train_losses = state["train_losses"]
            val_losses = state["val_losses"]
            self.learning_rate = state["learning_rate"]


            # restore weights and biases for each layer
            for layer, params in zip(self.layers, state["layer_params"]):
                layer.weights = params["weights"]
                layer.biases = params["biases"]

            print(f"Training state loaded from epoch {start_epoch + 1}")
            return start_epoch, train_losses, val_losses
        except FileNotFoundError:
            print("No saved state found. Starting from scratch.")
            return 0, [], []  # if no saved state, return starting values




