import numpy as np

from HW3.classes import Model, Training

input_size = 2
hidden_size = 3
output_size = 1

# Create a neural network model
model = Model(input_size, hidden_size, output_size)

# Sample training data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Train the model
Training.train(model, inputs, targets, learning_rate=0.1, epochs=10000)

# Make predictions
predictions = model.predict(inputs)
print("Predictions:")
print(predictions)