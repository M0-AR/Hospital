import pydicom
from pydicom.dataset import Dataset, FileDataset
from datetime import datetime
import tempfile
import os

def create_dicom_file(filename):
    # Create a new dataset and populate required values
    ds = Dataset()
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.Modality = "MR"
    ds.SeriesNumber = "1"
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.SeriesDate = datetime.now().strftime('%Y%m%d')
    ds.ContentDate = datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.now().strftime('%H%M%S')
    ds.SeriesTime = datetime.now().strftime('%H%M%S')
    ds.ContentTime = datetime.now().strftime('%H%M%S')
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Add SOPClassUID for MR Image Storage
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"

    # Save to DICOM file
    ds.save_as(filename)

if __name__ == "__main__":
    output_file = "/output/test.dcm"
    create_dicom_file(output_file)
    print(f"DICOM file saved to: {output_file}")
