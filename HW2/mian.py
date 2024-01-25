import numpy as np

from HW2.Perceptron import Perceptron
from HW2.test_perceptron import test_perceptron

input_size = 20 * 20  # Assuming 20x20 input images
bias = 1
perceptron = Perceptron(input_size, bias)

# Replace 'inputs' and 'targets' with your actual dataset
inputs = np.random.rand(100, input_size)  # Example: 100 images
targets = np.random.randint(2, size=(100, 1))  # Example: Binary labels

perceptron.train(inputs, targets, epochs=1000, learning_rate=0.01)


test_data = np.random.rand(50, input_size + 1)  # Example: 50 test samples
test_perceptron(perceptron, test_data)