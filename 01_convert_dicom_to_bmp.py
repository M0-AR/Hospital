import os
import pydicom
from PIL import Image
import os
import pydicom
import numpy as np
from PIL import Image
from collections import defaultdict


def convert_dicom_to_bmp(patient_folder, output_folder):
    # Dictionary to keep track of the count for each series
    series_counters = defaultdict(int)

    # This function converts all DICOM images in a patient's DICOM directory to BMP
    for root, dirs, files in os.walk(patient_folder):
        for file in files:
            # if file.lower().endswith('.dcm'):
            try:
                # Read the DICOM file
                dicom_path = os.path.join(root, file)
                ds = pydicom.dcmread(dicom_path)

                # Get patient, study and series information
                patient_id = ds.PatientID
                modality = ds.Modality 
                study_date = ds.StudyDate
                series_number = str(ds.SeriesNumber).zfill(4)
                # Create the BMP filename using a series-specific counter
                series_key = (ds.PatientID, ds.StudyDate, ds.SeriesNumber)
                series_counters[series_key] += 1
                counter_str = str(series_counters[series_key]).zfill(5)

                # Define output directory structure
                output_dir_structure = os.path.join(output_folder, f"Patient-{patient_id}", f"Study-{modality}[{study_date}]", f"Series-{series_number}")
                os.makedirs(output_dir_structure, exist_ok=True)

                # Get the pixel data from the DICOM dataset
                pixel_data = ds.pixel_array

                # Rescale the pixel data if Rescale Slope and Intercept are provided
                if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
                    pixel_data = pixel_data * ds.RescaleSlope + ds.RescaleIntercept

                # Apply windowing if Window Center and Width are provided
                if 'WindowCenter' in ds and 'WindowWidth' in ds:
                    window_center = ds.WindowCenter
                    window_width = ds.WindowWidth
                    
                    if isinstance(window_center, pydicom.multival.MultiValue):
                        window_center = window_center[0]
                    if isinstance(window_width, pydicom.multival.MultiValue):
                        window_width = window_width[0]
                    
                    lower_limit = window_center - window_width // 2
                    upper_limit = window_center + window_width // 2
                    pixel_data = np.clip(pixel_data, lower_limit, upper_limit)

                # Normalize the pixel data
                pixel_data = ((pixel_data - np.min(pixel_data)) / 
                            (np.max(pixel_data) - np.min(pixel_data))) * 255.0
                pixel_data = pixel_data.astype(np.uint8)

                # Convert the single-channel grayscale image to a 3-channel RGB image
                # pixel_data_rgb = np.stack((pixel_data,) * 3, axis=-1)
                pixel_data_rgb = np.stack((pixel_data,) * 4, axis=-1)

                # Convert to BMP and save
                image = Image.fromarray(pixel_data_rgb)
                bmp_filename = f"img-0{series_number}-{counter_str}.bmp"
                bmp_path = os.path.join(output_dir_structure, bmp_filename)
                image.save(bmp_path, 'BMP')
            except Exception as e:
                print(f"Error processing file {file}: {e}")


def convert_all_patients_to_bmp(dicom_data_folder, output_folder):
    # Walk through the dicom_data_folder recursively
    for root, dirs, files in os.walk(dicom_data_folder):
        if os.path.basename(root).upper() == 'DICOM':  # Checks if the current directory is named 'DICOM'
            patient_id = os.path.basename(os.path.dirname(root))  # Assumes the parent directory name is the patient ID
            print(f"Converting DICOM files for patient: {patient_id}")
            convert_dicom_to_bmp(root, output_folder)


# Example usage:
dicom_data_folder = 'C:/src/Canada/pipeline/02/dicom_data'
output_folder = 'C:/src/Canada/pipeline/02/bmp_data_by_python'
os.makedirs(output_folder, exist_ok=True)  # Create the output directory if it doesn't exist
convert_all_patients_to_bmp(dicom_data_folder, output_folder)
