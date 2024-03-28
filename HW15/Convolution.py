import numpy as np
from scipy.signal import convolve2d

def depthwise_convolution(image, kernel):

    output = np.zeros_like(image)
    for c in range(image.shape[2]):
        output[:, :, c] = convolve2d(image[:, :, c], kernel, mode='same', boundary='wrap')
    return output

def pointwise_convolution(image, kernel):

    if kernel.ndim != 1 or kernel.shape[0] != image.shape[2]:
        raise ValueError("Kernel must match the image's channels.")
    kernel = kernel.reshape((1, 1, -1))
    output = np.sum(image * kernel, axis=2, keepdims=True)
    return output

# Creating a synthetic image and kernels
image = np.random.rand(100, 100, 3)
depth_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
point_kernel = np.array([0.5, 0.25, 0.25])

# Choose the convolution type
flag = 'depthwise'  # Can be 'depthwise' or 'pointwise'

if flag == 'depthwise':
    result_image = depthwise_convolution(image, depth_kernel)
elif flag == 'pointwise':
    result_image = pointwise_convolution(image, point_kernel)
else:
    raise ValueError("Invalid flag. Choose either 'depthwise' or 'pointwise'.")