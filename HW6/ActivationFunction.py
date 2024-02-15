import numpy as np

class ActivationFunction:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return ActivationFunction.sigmoid(x) * (1 - ActivationFunction.sigmoid(x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def softmax_derivative(x):
        # softmax derivative is usually used in loss function, so we can just pass
        pass

# x = np.array([-1.0, 0.0, 1.0, 2.0])
# print("Linear:", ActivationFunction.linear(x))
# print("ReLU:", ActivationFunction.relu(x))
# print("Sigmoid:", ActivationFunction.sigmoid(x))
# print("Tanh:", ActivationFunction.tanh(x))
# softmax_x = np.array([[1, 2, 3, 6],
#                       [2, 4, 5, 6],
#                       [1, 2, 3, 6]])
# print("Softmax:", ActivationFunction.softmax(softmax_x))
