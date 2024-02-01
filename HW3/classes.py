import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()

    def activate(self, inputs):
        return Activation.sigmoid(np.dot(inputs, self.weights) + self.bias)

class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def activate(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Layer(input_size, hidden_size)
        self.output_layer = Layer(hidden_size, output_size)

    def predict(self, inputs):
        hidden_outputs = self.hidden_layer.activate(inputs)
        return self.output_layer.activate(hidden_outputs)

class LossFunction:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

class ForwardProp:
    @staticmethod
    def forward(model, inputs):
        hidden_outputs = model.hidden_layer.activate(inputs)
        return model.output_layer.activate(hidden_outputs), hidden_outputs

class BackProp:
    @staticmethod
    def backward(model, inputs, targets, learning_rate):
        output_layer_outputs, hidden_layer_outputs = ForwardProp.forward(model, inputs)

        output_errors = targets - output_layer_outputs
        output_delta = output_errors * Activation.sigmoid_derivative(output_layer_outputs)

        hidden_errors = output_delta.dot(model.output_layer.weights.T)
        hidden_delta = hidden_errors * Activation.sigmoid_derivative(hidden_layer_outputs)

        model.output_layer.weights += hidden_layer_outputs.T.dot(output_delta) * learning_rate
        model.output_layer.bias += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        model.hidden_layer.weights += inputs.T.dot(hidden_delta) * learning_rate
        model.hidden_layer.bias += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

class GradDescent:
    @staticmethod
    def update_weights(model, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            BackProp.backward(model, inputs, targets, learning_rate)

class Training:
    @staticmethod
    def train(model, inputs, targets, learning_rate, epochs):
        GradDescent.update_weights(model, inputs, targets, learning_rate, epochs)
        predictions = model.predict(inputs)
        loss = LossFunction.mean_squared_error(targets, predictions)
        print(f"Training Loss after {epochs} epochs: {loss}")
