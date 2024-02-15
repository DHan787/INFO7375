import numpy as np

from HW6.ActivationFunction import ActivationFunction


class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        self.output = None
        self.input = None
        self.dweights = None
        self.dbias = None

    def forward(self, input_data):
        self.input = input_data
        Z = np.dot(input_data, self.weights) + self.bias
        if self.activation == 'relu':
            self.output = ActivationFunction.relu(Z)
        else:
            self.output = Z  # Linear or no activation
        return self.output

    def backward(self, doutput):
        if self.activation == 'relu':
            dactivation = ActivationFunction.relu_derivative(self.output) * doutput
        else:
            dactivation = doutput

        self.dweights = np.dot(self.input.T, dactivation)
        self.dbias = np.sum(dactivation, axis=0, keepdims=True)
        dinput = np.dot(dactivation, self.weights.T)
        return dinput


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
        return np.mean(np.power(true_output - predicted_output, 2))

    def compute_loss_derivative(self, predicted_output, true_output):
        return 2 * (predicted_output - true_output) / true_output.size

    def backward_propagation(self, loss_derivative):
        for layer in reversed(self.layers):
            loss_derivative = layer.backward(loss_derivative)

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dweights
            layer.bias -= learning_rate * layer.dbias

    def fit(self, X_train, Y_train, epochs, learning_rate):
        for epoch in range(epochs):
            predicted_output = self.forward_propagation(X_train)
            loss = self.compute_loss(predicted_output, Y_train)
            self.loss_history.append(loss)

            loss_derivative = self.compute_loss_derivative(predicted_output, Y_train)
            self.backward_propagation(loss_derivative)
            self.update_parameters(learning_rate)

            print(f"Epoch {epoch + 1}, Loss: {loss}")


