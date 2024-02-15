import numpy as np

from HW6.ActivationFunction import ActivationFunction


class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        self.output = None
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        Z = np.dot(input_data, self.weights) + self.bias

        if self.activation is None:
            self.output = Z
        elif self.activation == 'relu':
            self.output = ActivationFunction.relu(Z)
        elif self.activation == 'softmax':
            self.output = ActivationFunction.softmax(Z)

        return self.output


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_history = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def compute_loss(self, predicted_output, true_output):
        # Mean Squared Error loss function
        return np.mean(np.power(true_output - predicted_output, 2))

    def fit(self, X_train, Y_train, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            predicted_output = self.forward_propagation(X_train)
            # Compute loss
            loss = self.compute_loss(predicted_output, Y_train)
            self.loss_history.append(loss)
            # Backpropagation and optimization would go here
            # For simplicity, it's not implemented in this snippet
            print(f"Epoch {epoch + 1}, Loss: {loss}")


