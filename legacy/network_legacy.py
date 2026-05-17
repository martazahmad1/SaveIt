import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    """Fully Connected Layer"""
    def __init__(self, input_size, output_size):
        super().__init__()
        # He initialization for weights (good for ReLU)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        # Initialize bias to zeros
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data):
        """
        Forward Pass
        Y = X * W + B
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Backward Pass (Gradient Descent)
        Calculates dE/dW, dE/dB and returns dE/dX for the previous layer
        """
        # Error with respect to the input of this layer
        input_error = np.dot(output_error, self.weights.T)
        
        # Error with respect to weights
        weights_error = np.dot(self.input.T, output_error)
        
        # Update weights and biases
        self.weights -= learning_rate * weights_error
        # Sum the output error over the batch for the bias
        self.bias -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        
        return input_error

class Activation(Layer):
    """Activation Layer"""
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        # Element-wise multiplication (Hadamard product)
        return self.activation_prime(self.input) * output_error

# --- Activation Functions ---

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1.0, 0.0)

def sigmoid(x):
    # Clip to avoid overflow/underflow
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

# --- Loss Function ---

class MSELoss:
    """Mean Squared Error Loss Function"""
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def loss_prime(y_true, y_pred):
        """Derivative of MSE with respect to predictions"""
        # dE/dY = 2 * (Y_pred - Y_true) / N
        return 2 * (y_pred - y_true) / y_true.size

# --- Autoencoder Framework ---

class Autoencoder:
    """Neural Network wrapper specifically for Unsupervised Autoencoding"""
    def __init__(self):
        self.layers = []
        self.loss_fn = MSELoss()

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        """Pass data sequentially through all layers"""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_error, learning_rate):
        """Propagate error backwards through all layers"""
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

    def fit(self, x_train, epochs, learning_rate, batch_size=32, verbose=True, callback=None):
        """
        Train the network on the provided dataset.
        Since it's an autoencoder, the target is the input itself (x_train == y_train).
        callback(msg): optional function to receive log messages in real-time.
        """
        n_samples = x_train.shape[0]
        history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Shuffle data for Stochastic Gradient Descent (SGD)
            indices = np.random.permutation(n_samples)
            x_train_shuffled = x_train[indices]
            
            # Process in mini-batches
            n_batches = 0
            for i in range(0, n_samples, batch_size):
                x_batch = x_train_shuffled[i:i+batch_size]
                y_batch = x_batch # Unsupervised: Output should reconstruct Input
                
                # Forward Pass
                output = self.forward(x_batch)
                
                # Compute Loss
                epoch_loss += self.loss_fn.loss(y_batch, output)
                n_batches += 1
                
                # Backward Pass (Calculus & weight updates)
                error = self.loss_fn.loss_prime(y_batch, output)
                self.backward(error, learning_rate)
                
            # Average loss over the epoch
            epoch_loss /= max(n_batches, 1)
            history.append(epoch_loss)
            
            msg = f"Epoch {epoch+1}/{epochs} | MSE Loss: {epoch_loss:.6f}"
            if verbose:
                print(msg)
            if callback:
                callback(msg)
                
        return history

    def get_weights(self):
        """Extracts weights from all Dense layers"""
        weights = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                weights.append({'weights': layer.weights, 'bias': layer.bias})
        return weights
        
    def set_weights(self, weights_list):
        """Restores weights to Dense layers"""
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.weights = weights_list[idx]['weights']
                layer.bias = weights_list[idx]['bias']
                idx += 1
