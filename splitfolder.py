import os
import shutil
import random

# Set the paths
dataset_folder = 'C:\\Users\\devra\\OneDrive\\Documents\\Projects\\Fed_ML\\Dataset'
output_folder = 'C:\\Users\\devra\\OneDrive\\Documents\\Projects\\Fed_ML\\OutputDataset'

# Create output directories if they don't exist
os.makedirs(output_folder, exist_ok=True)
part_folders = ['Part1', 'Part2', 'Part3']
for part in part_folders:
    os.makedirs(os.path.join(output_folder, part), exist_ok=True)

# List of subfolders in the dataset folder
subfolders = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

for subfolder in subfolders:
    subfolder_path = os.path.join(dataset_folder, subfolder)
    files = os.listdir(subfolder_path)
    
    # Shuffle the files randomly
    random.shuffle(files)
    
    # Distribute files into the three parts
    part_index = 0
    for file_name in files:
        source = os.path.join(subfolder_path, file_name)
        destination_subfolder = os.path.join(output_folder, part_folders[part_index], subfolder)
        os.makedirs(destination_subfolder, exist_ok=True)
        destination = os.path.join(destination_subfolder, file_name)
        shutil.move(source, destination)
        
        # Move to the next part index
        part_index = (part_index + 1) % 3

print("Dataset has been divided into three parts successfully.")