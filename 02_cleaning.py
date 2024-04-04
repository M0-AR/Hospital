import os
import shutil

# Initialize the global counter
global counter
counter = 1
counter = 27291

def copy_files(src_folder, dst_folder):
    global counter
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            # Generate source file path
            src_file_path = os.path.join(root, file)
            # Generate destination file path, using the counter for unique names
            dst_file_path = os.path.join(dst_folder, f"{counter}_{file}")
            # Copy the file to the destination folder
            shutil.copy2(src_file_path, dst_file_path)
            # Increment the counter after each file is copied
            counter += 1

def consolidate_data(base_dir, target_kidneys, target_no_kidneys):
    # Check for and create target directories if they don't exist
    if not os.path.exists(target_kidneys):
        os.makedirs(target_kidneys)
    if not os.path.exists(target_no_kidneys):
        os.makedirs(target_no_kidneys)

    for patient_folder in next(os.walk(base_dir))[1]:
        patient_path = os.path.join(base_dir, patient_folder)
        # Define source folders
        src_kidneys = os.path.join(patient_path, 'kidneys')
        src_no_kidneys = os.path.join(patient_path, 'no_kid_tum')
        
        # Copy files from source to target folders
        if os.path.exists(src_kidneys):
            copy_files(src_kidneys, target_kidneys)
        # if os.path.exists(src_no_kidneys):
        #     copy_files(src_no_kidneys, target_no_kidneys)
        print(f"Data from {patient_folder} has been consolidated.")

if __name__ == "__main__":
    base_dir = "G:\\Malignant_kidnokid"
    base_dir = "G:\\Benign_kidnokid"
    target_kidneys = "K:/kidney_classification_data/kidneys"
    target_no_kidneys = "K:/kidney_classification_data/no_kidneys"
    consolidate_data(base_dir, target_kidneys, target_no_kidneys)