from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from datetime import datetime

def create_dicom_file(filename):
    # Create file meta information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Create a new dataset and populate required values
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False  # Set to False because we're using explicit VR

    # Set the dataset specifics
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = "MR"
    ds.SeriesNumber = "1"
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.SeriesDate = datetime.now().strftime('%Y%m%d')
    ds.ContentDate = datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.now().strftime('%H%M%S')
    ds.SeriesTime = datetime.now().strftime('%H%M%S')
    ds.ContentTime = datetime.now().strftime('%H%M%S')

    # Add SOPClassUID for MR Image Storage
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    # Save to DICOM file with preamble
    ds.save_as(filename, write_like_original=False)

if __name__ == "__main__":
    output_file = "test.dcm"
    create_dicom_file(output_file)
    print(f"DICOM file saved to: {output_file}")
