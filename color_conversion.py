from PIL import Image
import os

# Define the directory
directory = 'MRIs'
output_directory = 'MRIs_greyscale'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.png'):  # Check for PNG files
        # Construct full file path
        file_path = os.path.join(directory, filename)

        # Open the image
        img = Image.open(file_path)

        # Convert to greyscale
        img_greyscale = img.convert('L')

        # Construct output file path
        output_file_path = os.path.join(output_directory, f'greyscale_{filename}')

        # Save the greyscale image
        img_greyscale.save(output_file_path)

        print(f'Converted {filename} to greyscale and saved as {output_file_path}')
