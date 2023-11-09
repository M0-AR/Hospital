from pynetdicom import AE, StoragePresentationContexts
from pynetdicom.sop_class import MRImageStorage
from pydicom import dcmread
from pydicom.fileset import FileSet

# Your Orthanc DICOM server details
orthanc_server_ip = '130.226.25.53'
orthanc_server_port = 4242

# The path to the DICOM file to send
dicom_file_path = './test.dcm'

# Initialise the Application Entity
ae = AE()

# Add a requested presentation context
ae.requested_contexts = StoragePresentationContexts

# Read the DICOM file
ds = dcmread(dicom_file_path)

# Create an association with the peer AE at IP 127.0.0.1 and port 4242
assoc = ae.associate(orthanc_server_ip, orthanc_server_port)

if assoc.is_established:
    # Use the C-STORE service to send the DICOM file
    status = assoc.send_c_store(ds)

    # Check the status of the storage request
    if status:
        # If the storage request succeeded, this should be 0x0000
        print('C-STORE request status: 0x{0:04x}'.format(status.Status))
    else:
        print('Connection timed out, was aborted or received invalid response')

    # Release the association
    assoc.release()
else:
    print('Association rejected, aborted or never connected')

