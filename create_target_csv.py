import os
import pandas as pd

# Define the base directory
base_directory = 'MRIs_greyscale'

# Iterate over all directories in the base directory
for subdir in os.listdir(base_directory):
    subdir_path = os.path.join(base_directory, subdir)
    if os.path.isdir(subdir_path):  # Check if it's a directory
        file_names = []
        # Iterate over all files in the subdirectory
        for filename in os.listdir(subdir_path):
            if filename.endswith('.tif'):  # Check for tif files
                file_names.append(filename)

        # Create a DataFrame with file names
        df = pd.DataFrame(file_names, columns=['File Name'])

        # Add empty columns for x and y coordinates
        df['X Coordinate'] = ''
        df['Y Coordinate'] = ''

        # Save the DataFrame to a CSV file within the subdirectory
        output_file = 'targets.csv'
        df.to_csv(os.path.join(subdir_path, output_file), index=False)

        print(f'Spreadsheet created and saved in {subdir_path} as {output_file}')
