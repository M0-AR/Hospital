# https://medium.com/@carlosrl/integrating-ai-into-clinical-workflow-with-orthanc-and-ohif-viewer-27bfc64f2718
import requests
from pydicom import dcmread
from io import BytesIO
from PIL import Image
import numpy as np

# DICOM Server details
server_url = 'http://127.0.0.1:8042'
auth = ('orthanc', 'orthanc')  # if required

# Function to get the list of studies
def get_studies():
    response = requests.get(f'{server_url}/studies', auth=auth)
    if response.status_code == 200:
        studies = response.json()
        return studies
    else:
        print(f"Failed to retrieve studies. Status code: {response.status_code}")
        # Print the response content that might contain the error message
        print(f"Response content: {response.content.decode('utf-8')}")
        return []

# Function to get the last study
def get_last_study():
    studies = get_studies()
    if not studies:
        return None

    # If 'studies' is a list of strings (study IDs), then return the last one
    last_study_id = studies[-1]  # Assuming the last one is the most recent
    return last_study_id

# Function to get a DICOM file
def get_dicom(study_id):
    response = requests.get(f'{server_url}/studies/{study_id}', auth=auth)
    if response.status_code == 200:
        try:
            # Attempt to read with the force option
            dicom_data = dcmread(BytesIO(response.content), force=True)
            return dicom_data
        except Exception as e:
            print(f"Failed to read DICOM data: {e}")
            return None
    else:
        print(f"Failed to retrieve data for study ID {study_id}")
        return None

print("######################################")
print("Get Last Study")
print("######################################")

# Example usage
study_id = get_last_study()

if study_id:
    dicom_file = get_dicom(study_id)

    # Check if the dicom_file is not None
    if dicom_file:
        print("Available tags in the DICOM file:")
        for element in dicom_file:
            # Print tag group and element numbers, tag name, and its value
            print(f"({element.tag.group:04x}, {element.tag.element:04x}) {element.name}: {element.value}")
        # Continue with further processing...
    else:
        print("Failed to read the DICOM file or no file was returned.")
else:
    print("No studies found")

print("\n\n\n")

print("######################################")
print("Get DICOM images for Last Study")
print("######################################")
# Function to get series from a study
def get_series_from_study(study_id):
    response = requests.get(f'{server_url}/studies/{study_id}/series', auth=auth)
    if response.status_code == 200:
        series_list = response.json()
        return series_list
    else:
        print(f"Failed to retrieve series for study {study_id}")
        return []

# Function to get instances from a series
def get_instances_from_series(series_id):
    response = requests.get(f'{server_url}/series/{series_id}/instances', auth=auth)
    if response.status_code == 200:
        instances_list = response.json()
        return instances_list
    else:
        print(f"Failed to retrieve instances for series {series_id}")
        return []

# Function to download a DICOM instance
def download_instance(instance_id):
    response = requests.get(f'{server_url}/instances/{instance_id}/file', auth=auth)
    if response.status_code == 200:
        # You can save the file or return its content
        return BytesIO(response.content)
    else:
        print(f"Failed to download instance {instance_id}")
        return None

def process_dicom(dicom_io):
    # Read the DICOM file from the BytesIO object
    ds = dcmread(dicom_io)

    # Assuming that the DICOM images are monochrome (grayscale)
    # Convert the pixel array to a PIL Image object
    image = Image.fromarray(ds.pixel_array.astype(np.uint8))

    # If necessary, apply windowing or other image processing here
    # ...

    # Save the image to a bytes buffer to prepare for the API request
    buffered = BytesIO()
    image_format = 'PNG'  # or 'JPEG' as required by the AI service
    image.save(buffered, format=image_format)
    
    # This is the binary data you will send to the AI service
    image_data = buffered.getvalue()

    # Here you can add the code to send the image_data to the AI service
    ai_service_url = 'http://ai-service.endpoint/api'
    response = requests.post(
        ai_service_url,
        files={'file': ('image.' + image_format.lower(), image_data, 'image/' + image_format.lower())}
    )

    # Check the response
    if response.status_code == 200:
        print('Successfully sent the DICOM image to the AI service.')
        # Process the AI service response as needed
    else:
        print('Failed to send the DICOM image to the AI service.')

# Get all series in the study
series_list = get_series_from_study(study_id)

if series_list:
    for series in series_list:
        series_id = series['ID']
        instances = get_instances_from_series(series_id)
        
        if instances:
            for instance in instances:
                instance_id = instance['ID']
                dicom_io = download_instance(instance_id)
                
                if dicom_io:
                    process_dicom(dicom_io)
else:
    print("No series found")