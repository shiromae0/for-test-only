from PIL import Image
import os

# Define the directory where the images are located (current directory)
directory_path = '.'

# Get list of image files in the directory that match the pattern
image_files = [f for f in os.listdir(directory_path) if f.startswith('belt_W_A_') and f.endswith('.png')]

# Define the new size for the images
new_size = (50, 50)

# Resize each image and save it
for image_file in image_files:
    image_path = os.path.join(directory_path, image_file)
    with Image.open(image_path) as img:
        resized_img = img.resize(new_size)
        resized_img.save(image_path)  # Overwrite the original file with the resized version

print(f"Resized images: {image_files}")
