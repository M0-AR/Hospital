import os
import zipfile
from pathlib import Path

def zip_folder(folder_path: Path, output_zip_path: Path):
    # Ensure the folder exists
    if not folder_path.exists() or not folder_path.is_dir():
        print("Invalid folder path.")
        return
    
    # Create a ZipFile object in write mode
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the folder
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                # Construct the full path
                file_path = Path(foldername) / filename
                
                # Create an archive name that is relative to the folder being zipped
                arcname = file_path.relative_to(folder_path)
                
                # Add the file to the zip file
                # The arcname parameter avoids storing the absolute path in the zip file
                zipf.write(file_path, arcname)

    print(f"Successfully zipped {folder_path} into {output_zip_path}")

# Example usage:
folder_to_zip = Path("C:/Users/UKF/Desktop/Renal-Cancer-Project/Regionsjaelland_Hospital_Benign_And_Malignant_Data")
output_zip_file = Path("Regionsjaelland_Hospital_CancerData.zip")
zip_folder(folder_to_zip, output_zip_file)
