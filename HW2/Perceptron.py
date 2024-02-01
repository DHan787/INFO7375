import numpy as np


class Perceptron:
    def __init__(self, input_size, input_number): # not input number, but the number of labels
        self.weights = np.random.rand(input_size, input_number)  # Initialize weights with random values
        self.bias = np.random.rand()  # Initialize bias with a random value

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def log_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0) or log(1)
        return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def forward(self, inputs):
        linear_output = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def backward(self, inputs, targets, learning_rate):
        predictions = self.forward(inputs)
        error = targets - predictions

        # Compute gradients
        weight_gradients = np.dot(inputs.T, error)
        bias_gradient = np.sum(error)

        # Update parameters using gradient descent
        self.weights += learning_rate * weight_gradients
        self.bias += learning_rate * bias_gradient

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            predictions = self.forward(inputs)
            loss = np.mean(self.log_loss(targets, predictions))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            self.backward(inputs, targets, learning_rate)

