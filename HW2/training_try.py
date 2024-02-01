import numpy as np
from PIL import Image
from _ctypes import sizeof
from sklearn.model_selection import train_test_split

from HW2.Perceptron import Perceptron


def preprocess_image(filename):
    image = Image.open(filename).convert('L')
    image = image.resize((20, 20))
    image_array = np.array(image)
    flattened_image = image_array.reshape((400,))
    normalized_image = flattened_image / 255.0
    return normalized_image


def load_and_preprocess_data(data_folder):
    inputs, labels = [], []

    for digit in range(10):
        for variation in range(10):
            filename = f"{data_folder}/digit_{digit}_variation_{variation}.png"
            image_vector = preprocess_image(filename)

            inputs.append(image_vector)
            labels.append(digit)

    inputs = np.array(inputs)
    labels = np.array(labels)

    return inputs, labels


# Example usage:
data_folder = "images/generated_grayscale"
inputs, labels = load_and_preprocess_data(data_folder)


# Split the data into training and testing sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)


# Initialize and train the perceptron
input_size = 20 * 20
input_number = 10
perceptron = Perceptron(input_size, input_number)
perceptron.train(train_inputs, train_labels, epochs=1000, learning_rate=0.01)

# Test the perceptron on the testing set
correct_predictions = 0
total_samples = len(test_inputs)

for i in range(total_samples):
    prediction = perceptron.forward(test_inputs[i])
    binary_prediction = 1 if prediction > 0.5 else 0

    if binary_prediction == test_labels[i]:
        correct_predictions += 1

accuracy = correct_predictions / total_samples
print(f"Accuracy on the testing set: {accuracy * 100}%")
