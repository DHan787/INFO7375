import numpy as np

def convolution(image, filter):
    # Define the shape of the output feature map
    output_shape = (image.shape[0] - filter.shape[0] + 1, image.shape[1] - filter.shape[1] + 1)

    # Initialize the output feature map
    output = np.zeros(output_shape)

    # Perform convolution
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            output[i, j] = np.sum(image[i:i+filter.shape[0], j:j+filter.shape[1]] * filter)

    return output

