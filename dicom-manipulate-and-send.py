import os
import pydicom
import shutil
import time
import numpy as np
import threading
import logging
import pynetdicom
from datetime import datetime
from pynetdicom import AE, StoragePresentationContexts
from pydicom.uid import ImplicitVRLittleEndian, ExplicitVRLittleEndian, JPEGBaseline, JPEGExtended
import configparser

# Get the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path of the configuration file
config_file_path = os.path.join(script_dir, "dicom-manipulate-and-send.ini")

# Load the configuration file
config = configparser.ConfigParser()
config.read(config_file_path)

# Constants and Configurations
source_dicom_folder = config.get("source", "source_dicom_folder")
local_ae_title = os.environ.get('COMPUTERNAME')
remote_ip = config.get("destination", "ip")
remote_port = int(config.get("destination", "port"))
remote_ae_title = config.get("destination", "ae_title")
USE_MULTITHREAD = config.get("options", "USE_MULTITHREAD")
MAX_CONCURRENT_THREADS = int(config.get("options", "MAX_CONCURRENT_THREADS"))
CLEANUP_OUTPUT_FOLDER = config.get("options", "CLEANUP_OUTPUT_FOLDER")
USE_SINGLE_ASSOCIATION = config.get("options", "USE_SINGLE_ASSOCIATION")

# Configure logging
logging.basicConfig(filename='dicom_sender.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set pynetdicom's logger to only display error messages
pynetdicom_logger = logging.getLogger('pynetdicom')
pynetdicom_logger.setLevel(logging.ERROR)


def determine_modality(dicom_folder):
    modalities_encountered = set()
    
    for dicom_filename in os.listdir(dicom_folder):
        dicom_path = os.path.join(dicom_folder, dicom_filename)
        ds = pydicom.dcmread(dicom_path)
        
        modality = ds.Modality
        if modality not in ["SR", "PR", "KO"]:
            modalities_encountered.add(modality)

    if len(modalities_encountered) == 1:
        return modalities_encountered.pop()
    else:
        return "MIXED"

primary_modality = determine_modality(source_dicom_folder)

def modify_pixel_data(image_data, percentage_change):
    modified_data = image_data * (1 + percentage_change)
    modified_data = np.clip(modified_data, 0, 2**16 - 1)
    return modified_data.astype(np.uint16)

def send_dicom_files(remote_ip, remote_port, remote_ae_title, local_ae_title, dicom_folder, use_multithread=USE_MULTITHREAD,
                     cleanup_output_folder=CLEANUP_OUTPUT_FOLDER, use_single_association=USE_SINGLE_ASSOCIATION):
    logging.info("Sending DICOM files started")

    # Set up the Association and Association parameters
    ae = AE(ae_title=local_ae_title)
    ae.requested_contexts = StoragePresentationContexts
    assoc_kwargs = {'addr': remote_ip, 'port': remote_port, 'ae_title': remote_ae_title}

    # Establish the Association
    assoc = None
    if use_single_association:
        assoc = ae.associate(**assoc_kwargs)

        if not assoc.is_established:
            logging.error("Association not established")
            return

    # Create a semaphore to limit the number of concurrent threads
    semaphore = threading.Semaphore(MAX_CONCURRENT_THREADS)

    def send_file(file_path):
        logging.info(f"Attempting to send file: {file_path}")  # Log which file is being sent

        try:
            ds = pydicom.dcmread(file_path)
            status = assoc.send_c_store(ds)
            if status:
                logging.info(f"Successfully sent {file_path}")
            else:
                logging.error(f"Failed to send {file_path}")
        except Exception as e:
            logging.error(f"Error sending {file_path}: {e}")

    # Threaded send function that utilizes the semaphore
    def threaded_send(file_path):
        with semaphore:
            send_file(file_path)

    # Get a list of DICOM files to send
    dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]

    if use_multithread:
        threads = []
        for dicom_file in dicom_files:
            file_path = os.path.join(dicom_folder, dicom_file)
            thread = threading.Thread(target=threaded_send, args=(file_path,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()
    else:
        for dicom_file in dicom_files:
            file_path = os.path.join(dicom_folder, dicom_file)
            send_file(file_path)

    # Release the Association if using a single association
    if use_single_association:
        assoc.release()

    logging.info("Sending DICOM files finished")

    # Cleanup the output folder if specified
    if cleanup_output_folder:
        try:
            shutil.rmtree(dicom_folder)
            logging.info(f"Successfully removed the output folder: {dicom_folder}")
        except Exception as e:
            logging.error(f"Error removing the output folder: {e}")




# Duplicate and modify DICOM files
duplicate_output_folder = f'output_{int(time.time())}'
os.makedirs(duplicate_output_folder, exist_ok=True)

start_time = time.time()
start_time_formatted = datetime.fromtimestamp(start_time).strftime('%Y%m%d%H%M%S')
current_date_formatted = datetime.now().strftime('%Y%m%d')
common_study_uid = pydicom.uid.generate_uid()

series_numbers = {}

for dicom_filename in os.listdir(source_dicom_folder):
    dicom_path = os.path.join(source_dicom_folder, dicom_filename)
    ds = pydicom.dcmread(dicom_path)

    series_number = ds.SeriesNumber
    if series_number not in series_numbers:
        series_numbers[series_number] = pydicom.uid.generate_uid()
    
    ds.SeriesInstanceUID = series_numbers[series_number]
    ds.StudyInstanceUID = common_study_uid
    ds.PatientID = os.environ.get('COMPUTERNAME', 'UnknownComputer')
    ds.PatientBirthDate = current_date_formatted
    ds.PatientName = os.environ.get('COMPUTERNAME', 'UnknownComputer')
    ds.StationName = os.environ.get('COMPUTERNAME', 'UnknownComputer')
    # Wipe out specified fields
    fields_to_delete = [
        "PerformingPhysicianName",                          # (0008,1050)
        "ReferringPhysicianName",                           # (0008,0090)
        "ConsultingPhysicianName",                          # (0008,009C)
        "PhysiciansOfRecord",                               # (0008,1048)
        "RequestingPhysician",                              # (0032,1032)
        "IssuerOfPatientID",
        "OperatorsName",
        "OtherPatientIDs",
        "ReasonForTheImagingServiceRequest",
        "OtherPatientIDsSequence",
        "ReasonForTheRequestedProcedure",                   # (0040,1002)
        "ReasonForStudy",               
        "StudyID",                                          # (0020,0010)
        "FillerOrderNumberImagingServiceRequest",           # (0040,2017)
        "RequestedProcedureCodeSequence",                   # (0032,1064)
        "InstitutionAddress",                               # (0008,0081)
        "InstitutionName",                                  # (0008,0080)
        "ScheduledProcedureStepID",                         # (0040,0009)
        "RequestedProcedureID",                             # (0040,1001)
        "InstitutionalDepartmentName",                      # (0008,1040)
        "RetrieveAETitle",                                  # (0008,0054)	
        "ProcedureCodeSequence",                            # (0008,1032)		
        "RequestAttributesSequence"                         # (0040,0275)	
    ]   
    for field in fields_to_delete:
        if field in ds:
            delattr(ds, field)     
    # Set the Procedure Description
    ds.StudyDescription = primary_modality + " Testing Images"
    ds.RequestedProcedureDescription = primary_modality + " Testing Images"
    
    # Ensure pixel_array is defined before entering the conditions
    pixel_array = None
    if ds.file_meta.TransferSyntaxUID in [JPEGBaseline, JPEGExtended]:
        # Change the Transfer Syntax to Implicit VR Little Endian
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        try:
            pixel_array = ds.pixel_array  # This will decompress the data
        except Exception as e:
            logging.error(f"Failed to read pixel_array for {dicom_path}: {e}")
            continue
    else:
        try:
            pixel_array = ds.pixel_array
        except Exception as e:
            logging.error(f"Failed to read pixel_array for {dicom_path}: {e}")

    if pixel_array is None:
        logging.error(f"Skipping due to undefined pixel_array for {dicom_path}")
        continue  # Skip the current iteration and move to the next DICOM file

    modified_pixel_array = modify_pixel_data(pixel_array, 0.05)
    ds.PixelData = modified_pixel_array.tobytes()
    
    # Ensure Pixel Data is not encapsulated
    if hasattr(ds, 'PixelData') and isinstance(ds.PixelData, list):
        ds.PixelData = b''.join(ds.PixelData)

    # Update other necessary tags
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0

    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.AccessionNumber = start_time_formatted
    print(f'Accession Number: {start_time_formatted}')
    # Ensure the filename ends with .dcm
    if not dicom_filename.lower().endswith('.dcm'):
        dicom_filename = f"{dicom_filename}.dcm"
        
    output_dicom_filename = os.path.join(duplicate_output_folder, dicom_filename)
    if ds.file_meta.TransferSyntaxUID not in [JPEGBaseline, JPEGExtended]:  # If not JPEG compressed
        ds.is_implicit_VR = True
        ds.is_little_endian = True
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.save_as(output_dicom_filename, write_like_original=True)
    
print("Modified and anonymized DICOM files have been generated and saved.")

# Calculate the size of the source DICOM folder in MB
source_folder_size_mb = sum(os.path.getsize(os.path.join(source_dicom_folder, f)) for f in os.listdir(source_dicom_folder)) / (1024 * 1024)


# Record the start time of DICOM send
send_start_time = time.time()

# Send DICOM files using multiple threads
send_dicom_files(remote_ip, remote_port, remote_ae_title, local_ae_title, duplicate_output_folder, use_multithread=USE_MULTITHREAD, cleanup_output_folder=CLEANUP_OUTPUT_FOLDER)

# Calculate the time taken for DICOM send
send_end_time = time.time()
send_duration = send_end_time - send_start_time

logging.info("Sending DICOM files finished")
logging.info(f"Time taken for DICOM send to {remote_ip}: {send_duration:.2f} seconds")
print(f"Time taken for DICOM send to {remote_ip}: {send_duration:.2f} seconds")
# Calculate Mbps based on source folder size and send duration
mbps = (source_folder_size_mb * 8) / send_duration
logging.info(f"Mbps: {mbps:.2f}")
print(f"Mbps: {mbps:.2f}")