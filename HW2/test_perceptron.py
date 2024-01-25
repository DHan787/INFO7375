import os

import numpy as np
from PIL import Image

from HW2.Perceptron import Perceptron

# this file is unusable, using this file to train will cause the loss to be negative
def preprocess_image(filename):
    # Load image and convert it to grayscale
    image = Image.open(filename).convert('L')

    # Resize the image to 20x20 pixels
    image = image.resize((20, 20))

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Flatten the image array to a 1D vector
    flattened_image = image_array.flatten()

    # Normalize pixel values to the range [0, 1]
    normalized_image = flattened_image / 255.0

    return normalized_image


def predict_with_perceptron(perceptron, image_filename):
    # Preprocess the image
    test_image = preprocess_image(image_filename)

    # Forward pass through the perceptron
    prediction = perceptron.forward(test_image)

    # Convert the prediction to a binary value
    binary_prediction = 1 if prediction > 0.5 else 0

    return binary_prediction

def load_data(image_folder):
    inputs, targets = [], []

    for digit in range(10):
        for variation in range(10):
            filename = os.path.join(image_folder, f"digit_{digit}_variation_{variation}.png")
            image_vector = preprocess_image(filename)

            inputs.append(image_vector)
            targets.append(digit)

    inputs = np.array(inputs)
    targets = np.array(targets).reshape(-1, 1)

    return inputs, targets


input_size = 20 * 20
bias = np.random.rand(1)
perceptron = Perceptron(20*20, bias)
training_folder = "images/generated_grayscale/"
training_inputs, training_targets = load_data(training_folder)
learning_rate = 0.01
epochs = 1000
perceptron.train(training_inputs, training_targets, epochs, learning_rate)

# Replace 'image_folder' with the folder where your ten images are stored
image_folder = "test_grayscale/"


