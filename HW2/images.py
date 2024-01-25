import numpy as np
from PIL import Image
from PIL import Image, ImageDraw
import os


# tried to convert the images to grayscale, but it didn't work, the converted images are blurry
def convert_to_grayscale(input_folder, output_folder):
    # List all files in the input folder
    file_list = os.listdir(input_folder)

    for filename in file_list:
        if filename.endswith(".png"):
            # Construct the file paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image file
            original_image = Image.open(input_path)

            # Convert the image to grayscale
            grayscale_image = original_image.convert('L')

            # Resize the image to 20x20 pixels
            resized_image = grayscale_image.resize((20, 20))

            # Save the resized grayscale image
            resized_image.save(output_path)


def generate_images_with_variations(output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for digit in range(10):
        for variation in range(10):
            # Create a 20x20 grayscale image
            image = Image.new('L', (20, 20), color=255)
            draw = ImageDraw.Draw(image)

            # Add variation to the position and style of the digit
            x_offset = np.random.randint(-2, 3)
            y_offset = np.random.randint(-2, 3)
            font_size = np.random.randint(10, 15)

            # Vary the appearance of the numbers using different styles, rotations, etc.
            draw.text((3 + x_offset, 3 + y_offset), str(digit), fill=0, font=None, fontsize=font_size)

            # Save the image with a filename that includes the digit and variation
            filename = os.path.join(output_folder, f"digit_{digit}_variation_{variation}.png")
            image.save(filename)


input_folder = "images/test/"
output_folder = "images/test_grayscale/"
convert_to_grayscale(input_folder, output_folder)
# output_folder = "images/generated_grayscale/"
# generate_images_with_variations(output_folder)



