import os
from PIL import Image

def split_image(image_path):
    with Image.open(image_path) as img:
        width, height = img.size

        # Cut the image in half
        left_half = img.crop((0, 0, width//2, height))
        right_half = img.crop((width//2, 0, width, height))

        # Save each half
        left_half.save(image_path.replace('.bmp', '_left_half.bmp'))
        right_half.save(image_path.replace('.bmp', '_right_half.bmp'))

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.bmp'):
                image_path = os.path.join(root, file)
                split_image(image_path)
                os.remove(image_path)  # Remove the original image

# Replace the directory path with your actual folder path
process_directory('H:/cut_in_half_Benign_and_Malignant_tumors_images')
