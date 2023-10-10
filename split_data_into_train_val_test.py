"""
To split your dataset into training, validation, and test sets and organize images and labels into corresponding folders, you may follow the below steps:

Use os and shutil to create directories and move files.
Use glob to retrieve all image and label files.
Use random to shuffle and split the data into train, validation, and test sets.

Notes:

os.makedirs: ensures the directories are created.
The script assumes you have .png image files and corresponding .txt label files.
random.shuffle: randomizes the order of the files before splitting them.
The script moves the files into a structure suitable for machine learning frameworks like TensorFlow and PyTorch.
Splitting ratios (70% training, 15% validation, 15% testing) are quite common, but you might adjust them according to your use case.
Make sure that each image file has a corresponding label file with the exact same name before running the script.
Always double-check manually if some of the files are moved correctly and datasets are split as expected. This will ensure that you don’t train/test your model on incorrect data.
Be careful with file moving and deleting operations, make sure to have a backup of your data.
"""

import os
import shutil
import random

def copy_files(file_list, dest_folder):
    for file in file_list:
        img_src = os.path.join(base_dir, 'images', file + '.png')
        lbl_src = os.path.join(base_dir, 'images', file + '.txt')
        
        try:
            shutil.copy(img_src, os.path.join(dest_folder, 'images'))
            shutil.copy(lbl_src, os.path.join(dest_folder, 'labels'))
        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")  # Debug line to check which file is not found

# Folders setup
base_dir = './'
output_dir = './splitted_data'

train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'valid')
test_dir = os.path.join(output_dir, 'test')

dirs = [train_dir, val_dir, test_dir]

subfolders = ['images', 'labels']

# Create directories
for dir in dirs:
    for subfolder in subfolders:
        os.makedirs(os.path.join(dir, subfolder), exist_ok=True)

# Fetch file names (without extension) of all images
all_files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(base_dir, 'images')) if f.endswith('.png')]

# Optionally set a seed for reproducibility
use_random_splitting = True  # Set to True or False
seed = 42  # Choose any number

if use_random_splitting:
    random.seed(seed)  # Set the seed
    random.shuffle(all_files)

# Adjusted ratios
num_train = int(0.8 * len(all_files))  # 80% for training
num_val = int(0.1 * len(all_files))    # 10% for validation

train_files = all_files[:num_train]
val_files = all_files[num_train:num_train + num_val]
test_files = all_files[num_train + num_val:]

# Copy files into corresponding directories
copy_files(train_files, train_dir)
copy_files(val_files, val_dir)
copy_files(test_files, test_dir)
