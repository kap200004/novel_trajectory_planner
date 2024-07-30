from PIL import Image
import os

# Define the directories
base_directory = 'MRIs'
output_base_directory = 'MRIs_greyscale'

# Create output base directory if it doesn't exist
if not os.path.exists(output_base_directory):
    os.makedirs(output_base_directory)

# Iterate over all directories in the base directory
for subdir in os.listdir(base_directory):
    subdir_path = os.path.join(base_directory, subdir)
    if os.path.isdir(subdir_path):  # Check if it's a directory
        output_subdir_path = os.path.join(output_base_directory, subdir)

        # Create corresponding directory in the output directory if it doesn't exist
        if not os.path.exists(output_subdir_path):
            os.makedirs(output_subdir_path)

        # Iterate over all files in the subdirectory
        for filename in os.listdir(subdir_path):
            if filename.endswith('.tif'):  # Check for PNG files
                # Construct full file path
                file_path = os.path.join(subdir_path, filename)

                # Open the image
                img = Image.open(file_path)

                # Convert to greyscale
                img_greyscale = img.convert('L')

                # Construct output file path
                output_file_path = os.path.join(output_subdir_path, filename)

                # Save the greyscale image
                img_greyscale.save(output_file_path)

                print(f'Converted {file_path} to greyscale and saved as {output_file_path}')
