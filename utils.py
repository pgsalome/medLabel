import subprocess as sp
import glob
import os
import warnings
import shutil
from pathlib import Path
import pydicom
import SimpleITK as sitk
import re
import csv
from converter import DicomConverters
import numpy as np
import nrrd
import nibabel as nib
from scipy import ndimage
from skimage.transform import resize
import fnmatch
#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import dicom2nifti
import dicom2nifti.settings as settings
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, OrderedDict 
from multiprocessing import Pool, cpu_count
from tqdm import tqdm 
import gc
import random

from scipy.ndimage import zoom
#### comments to do
## some ADC maps are appended to the last volume in the DWI 4D scan. write a script that looks for these and extracts them
# maybe if you have echo varying and image postion then it might be a indication
## for the 4d scans we can not rely only on ['Trigger Time ((0018, 1060))'] since sometimes  for 3d it is showing
# maybe extra  cheks of the sd if trigger time found to vary or look at the imagepoistion
# when training we can put attention on areas to help the mr classifier.e.g between pd and t2* attetion on the skull
# for t1fl and t2fl attention on the neck area bacuse of the fat
modality_considered = ["PT","CT","MR","DWI","PER","CB"]

#### string manip ####

def handle_none(value):
    if value is None:
        return 'NS'  # return an empty string if the value is None
    else:
        return value.replace("_","")
    

def split_string(s, phrases_to_keep= ['DIFFUSION TENSOR',"PROJECTION IMAGE","MR gesamte WirbelsÃ¤ul","gesamte WirbelsÃ¤ul"]):

    # Create a unique placeholder pattern for the phrases to keep together
    placeholders = {}
    for idx, phrase in enumerate(phrases_to_keep):
        placeholder = f"PHRASEPLACEHOLDER{idx}"
        placeholders[placeholder] = phrase
        s = s.replace(phrase, placeholder)
    
    # Now split the string based on the given delimiters
    parts = re.split('[ _\-/,\.]', s)

    # Replace back the placeholders with the original phrases
    for placeholder, phrase in placeholders.items():
        parts = [part.replace(placeholder, phrase) for part in parts]

    return [part for part in parts if part]

def remove_special_chars_and_numbers(s,remove_space = True):
    if remove_space:
        return re.sub('[^a-zA-Z*]+', '', s)
    else:
        return re.sub('[^a-zA-Z*]+', ' ', s)

def strip_non_ascii(string):
    return ''.join(i for i in string if ord(i) < 128)

#### image preprocessing ####

def convert_dicom_sitk(input_folder):

    # Get list of DICOM files
    try:
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_folder)
    
        # Read the DICOM series
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    
        # Write the image in NIfTI format
        sitk.WriteImage(image, input_folder+ '.nii.gz')
        return None
    except:
        return input_folder



def check_transfer_syntax_if_compress(dicom_file_path):
    """
    Check if the DICOM file's transfer syntax is not in the list of non-problematic syntaxes.
    If it is not, it might need to be decompressed.

    :param dicom_file_path: Path to the DICOM file
    :return: Boolean indicating whether the file needs decompression
    """
    # List of non-problematic (typically uncompressed) transfer syntaxes
    NotCompressedPixelTransferSyntaxes = [
        "1.2.840.10008.1.2.1",   # ExplicitVRLittleEndian
        "1.2.840.10008.1.2",     # ImplicitVRLittleEndian
        "1.2.840.10008.1.2.1.99",# DeflatedExplicitVRLittleEndian
        "1.2.840.10008.1.2.2"    # ExplicitVRBigEndian
    ]

    try:
        # Load the DICOM file
        ds = pydicom.dcmread(dicom_file_path)

        # Check if the TransferSyntaxUID is in the list of non-problematic syntaxes
        if hasattr(ds, 'file_meta') and ds.file_meta.TransferSyntaxUID not in NotCompressedPixelTransferSyntaxes:
            return True  # File might need decompression
        else:
            return False  # File does not need decompression
    except Exception as e:
        print(f"Error reading DICOM file: {e}")
        return False

    

def decompress_CT(image):
    # Check if the input is a list
    if isinstance(image, list):
        files = image
    else:
        # If it's not a list, assume it's a single path string
        files = glob.glob(image + '/*')
    
    # Process each file in the list
    for i in files:
        cmd = "gdcmconv --raw \"{0}\" \"{1}\" ".format(i, i)
        sp.check_output(cmd, shell=True)

def plastimatch_ct(image_dcm,outname):
    if not os.path.isfile(outname):
        cmd = ("plastimatch convert --input \"{0}\" --output-img \"{1}\" --output-type float" .format(image_dcm,outname))
        sp.check_output(cmd, shell=True) 
        
def fix_zero_spacing(input_image_path,  non_zero_value=1.0):
    # Read the image
    image = sitk.ReadImage(input_image_path)
    # Get the current spacing
    spacing = list(image.GetSpacing())
    # Check if any spacing value is zero, and replace it with a non-zero value
    for i, value in enumerate(spacing):
        if value == 0:
            spacing[i] = non_zero_value
    # Set the new spacing
    image.SetSpacing(spacing)
    # Save the modified image
    sitk.WriteImage(image, input_image_path)       

def convert_dicom2nifti(failed_ll,ext=".nii.gz"):

    for i,image in enumerate(failed_ll):
        try:
            if not os.path.isfile(image+ext):
                print(str(i)+'_'+image)
                dicom2nifti.dicom_series_to_nifti(image,image+ext,reorient_nifti=False)
        except:
            print("failed"+image)


######## for ploting ####


# def get_middle_slice(folder_path,convert_failed, nii_convert = True, view = "axial"):
#     """
#     Reads a DICOM image from a folder and returns the slice with the most information
#     and its series description DICOM attribute.
#     Args:
#     - folder_path (str): Path to the folder containing the DICOM file.
#     Returns:
#     - max_slice (numpy.ndarray): The slice with the most information.
#     - series_description (str): The DICOM series description attribute for the slice with the most information.
#     """
#     # convert to nrrd
#     #decompress_CT(folder_path)
#     # Get the middle slice
#     dcm_file = glob.glob(folder_path + '/*')[0]
#     ds = pydicom.dcmread(dcm_file)
#      # Get the DICOM series description attribute
#     try:
#         series_description = re.sub(r'[^\x00-\x7F\w\s]+', '', ds.SeriesDescription)
#     except:
#         series_description = ""
#     try:
#         modality =  ds.Modality
#     except:
#         series_description = ""
#     try:
#         contrast = ds.ContrastBolusAgent
#         contrast = "C"
#     except:
#         contrast = ""
#     if contrast == "C":
#         if modality !="PR":
#             modality = "C"+modality
        
#     if not nii_convert:
#         image = os.path.dirname(folder_path) +'/'+os.path.basename(folder_path)+'.nrrd'
#     else:
#         image = os.path.dirname(folder_path) +'/'+os.path.basename(folder_path)+'.nii.gz'
#     try:
#         if not nii_convert:
#             plastimatch_ct(folder_path,image)
#             image_data, header = nrrd.read(image)
#             #image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) 
#         else:
#             if not os.path.isfile(image):
#                 converter = DicomConverters(folder_path, ext='.nii.gz')
#                 status = converter.dcm2niix_converter(compress=True)
#                 if status is not None:
#                     convert_failed.append(status)
#             image_data = nib.load(image).get_fdata()
#             image_data = image_data.astype(np.float64)
#             #image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) 
# #            if image_data.dtype.names is not None:  # In case of structured data type
# #                r_data = image_data['R'].astype(np.float64)
# #                g_data = image_data['G'].astype(np.float64)
# #                b_data = image_data['B'].astype(np.float64)
# #                # Combine the channels
# #                image_data = np.stack((r_data, g_data, b_data), axis=-1)
# #            else:
# #                image_data = image_data.astype(np.float64)            
#     except:
#         try:
#             fix_zero_spacing(image)
#             if not nii_convert:
#                 image_data, header = nrrd.read(image)
#             else:
#                 image_data = nib.load(image).get_fdata()
#                 image_data = image_data.astype(np.float64)
#         except:
#             image_data = np.zeros((256,256,256))
#             orientation = ""
#             print(f"{folder_path} could not be converted even after trying zero spacing fix.") 
#     #
#     #        
#     if len(image_data.shape)>3:
#         image_data = image_data[:,:,:,0]
#         series_description = series_description +'_4D-I'    
    
#     orientation = get_scan_orientation(dcm_file)

#     #try:

#     if view == "coronal":
#         if orientation == "SAG":
#             image_data = image_data[image_data.shape[0]//2,:,:]
#         elif orientation == "TRA" or orientation == "3D" or orientation == "NA":
#             image_data = image_data[:,image_data.shape[1]//2,:]
#         else:
#             image_data = image_data[:,:,image_data.shape[2]//2]      
#     elif view == "axial":
#         if orientation == "SAG":
#             image_data = image_data[:,image_data.shape[1]//2,:]
#         elif orientation == "TRA" or orientation == "3D" or orientation == "NA" :
#             image_data = image_data[:,:,image_data.shape[2]//2]
#         else:
#             image_data = image_data[:,image_data.shape[1]//2,:]    
#     elif view == "sagittal":
#         if orientation == "SAG":
#             image_data = image_data[:,:,image_data.shape[2]//2]
#         elif orientation == "TRA" or orientation == "3D" or orientation == "NA":
#             image_data = image_data[image_data.shape[0]//2,:,:]
#         else:
#             image_data = image_data[image_data.shape[0]//2,:,:]      
#     elif view == "scan":
#         image_data = image_data[:,:,image_data.shape[2]//2]
        
            
#     # except:
#     #     print(image)
#     #     print(orientation)
#     #     print(image_data.shape)

    
#     return image_data, series_description, orientation, modality, contrast, convert_failed

def process_dicom_folders(dicom_folders, nrrd_convert, view):
    images = []
    series_descriptions = []
    orientations = []
    modalities = []
    contrasts = []
    convert_failed = []
    
    # Loop through all DICOM folders
    for folder in dicom_folders:
        # Call the get_middle_slice function to get the middle slice and associated metadata
        image, series_description, orientation, modality, contrast, convert_failed = get_middle_slice(folder,convert_failed, nrrd_convert, view)
        series_description = series_description.replace('_','')
        #series_description = series_description[:20] if len(series_description) > 20 else series_description
        image = ndimage.rotate(resize(image, (256, 256),order=3, mode='edge', cval=0, anti_aliasing=False),180)
        
        # Append the results to their respective lists
        images.append(image.astype(np.float32))
        series_descriptions.append(series_description)
        orientations.append(orientation)
        modalities.append(modality)
        contrasts.append(contrast)
    
    # Return the four lists
    return images, series_descriptions, orientations, modalities, contrasts, convert_failed


######## end for ploting ####

        
#### file manip ####



def rename_files_numerically(directory):
    # Get all DICOM files in the directory
    dicom_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.dcm')]

    # Sort files by their current name (assuming that reflects the slice location)
    n = dicom_files[0].split('/')[-1].split('.dcm')[0]
    if n.count('.')>1:
        dicom_files.sort(key=lambda x: float(os.path.basename(x).split('.dcm')[0].replace('.','')))
    else:
         dicom_files.sort(key=lambda x: float(os.path.basename(x).split('.dcm')[0]))

    # Rename files in numerical order
    for i, file_path in enumerate(dicom_files, start=1):
        new_name = os.path.join(directory, str(i) + '.dcm')
        os.rename(file_path, new_name)
       




def create_copy_toDir(file, dirName, new_filename=None):
    if not os.path.exists(dirName) and not os.path.isdir(dirName):
        os.makedirs(dirName, exist_ok=True)
        
    if new_filename:    
        destination = os.path.join(dirName, new_filename)
        base, extension = os.path.splitext(destination)
        counter = 1
        while os.path.exists(destination):
            destination = f"{base}_{counter}{extension}"
            counter += 1
    else:
        destination = dirName

    shutil.copy2(Path(file), destination)
    
def copy_file(args):
    #### reaplce  parts[subject_id_position] with key.split('-')[0] if name from dicom set
    #random_number = random.randint(1000, 9999)
    key, file_parts, output_dir, subject_id_position, scans_keys_index = args
    file = file_parts.split('_sloc')[0]
    sliceloc = file_parts.split('_sloc')[1]
    
    parts = file.split('/')
    dirName = os.path.join(output_dir, parts[subject_id_position],
                           key.split('-')[1], key.split('-')[2],
                           key.split('-')[3], key.split('-')[4], key.split('-')[5]+'_'+key.split('-')[6]+'_'+str(scans_keys_index))
    new_filename = None
    if sliceloc != "NS":
        new_filename = sliceloc + os.path.splitext(file)[1]    

    create_copy_toDir(file, dirName, new_filename)   

def delete_small_directories(root_dir, minimun_slices = 9,modality_considered=modality_considered):
    """
    Deletes directories with less than 30 .dcm files in the last level of subdirectories in the given root directory.
    """
    # ignore_list = ['UNCONVERTABLE','RTDOSE', 'RTSTRUCT', 'RTPLAN', 'CR', 'RTRECORD', 'RTIMAGE','SR','PR','REG','KO','US','NM','ECG']
    
    folders = [x for x in glob.glob(root_dir + '/*/*/*/*/*/*') if 'RTSTRUCT' not in x and os.path.isdir(x) and any(word in x.split('/')[-2] for word in modality_considered if word in x.split('/')[-2])]
    
    for folder in folders:  
        files = glob.glob(folder+'/*')
        if (len(files) < minimun_slices ):
      
            print(f"Deleting directory {folder} with {len(files)} dcm files")
            for file in files:
                os.remove(file)
            os.rmdir(folder)
            
def print_all_attributes(ds, indent=0):
    for element in ds.iterall():
        if element.VR == "SQ":  # sequence
            print(" " * indent + str(element.tag) + " " + element.name)
            for item in element.value:
                print_all_attributes(item, indent + 4)
        else:
            print(" " * indent + str(element.tag) + " " + element.name + " " + str(element.value))


def merge_folders(src_folder, dst_folder):
    for src_dir, dirs, files in os.walk(src_folder):
        dst_dir = src_dir.replace(src_folder, dst_folder, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir,exist_ok=True)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if not os.path.exists(dst_file):
                shutil.move(src_file, dst_dir)


def delete_empty_dirs(root_dir):
    # Use os.walk to generate a list of all directories in root_dir, in bottom-up order
    folder_found = False
    
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
 
        # If a directory contains no files or subdirectories, remove it
        dcm_files = [f for f in filenames if f.endswith('.dcm')]
        if not dirnames and not dcm_files:
            try:
                shutil.rmtree(dirpath)
                folder_found = True
                print(f"Removed empty directory: {dirpath}")
            except OSError as e:
          
                print(f"Error removing {dirpath}: {e}")
    
    return folder_found

def handle_files(root_dir, suffix=".dcm", action="keep"):
    # Check if action is either 'delete' or 'keep'
    if action not in ['delete', 'keep']:
        raise ValueError("Invalid action. Action should be either 'delete' or 'keep'")

    # Walk through root directory
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if '.csv' not in filename:
                # Check if file ends with the given suffix
                if filename.endswith(suffix):
                    file_path = os.path.join(foldername, filename)
                    if action == 'delete':
                        print(f"Deleting file: {file_path}")
                        os.remove(file_path) # delete the file
                    # If the action is 'keep', do nothing
                elif action == 'keep': 
                    file_path = os.path.join(foldername, filename)
                    print(f"Deleting file: {file_path}")
                    os.remove(file_path) # delete the file if it doesn't match the suffix   

def move_folders_to_new_dir(folders, new_dir_name):
    if not folders:
        print("No folders to move.")
        return

    
    for folder in folders:
        old_dir_name = os.path.basename(os.path.dirname(folder))
        if old_dir_name == new_dir_name:
            continue
        parent_dir = os.path.dirname(os.path.dirname(folder))
        
        new_dir = os.path.join(parent_dir, new_dir_name)

        # Create the new directory if it doesn't exist
        if not os.path.exists(new_dir):
            try:
                os.makedirs(new_dir,exist_ok=True)    
            except:
                pass
        # Construct new folder path
        new_folder_path = os.path.join(new_dir, os.path.basename(folder))
        
        # Move the folder
        try:
            shutil.move(folder, new_folder_path)
        except:
            pass



    


def compare_and_move(source_dir, target_dir, new_dir):
    # Create new directory if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Get the list of all directories in source and target directory
    source_subdirs = {d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))}
    target_subdirs = {d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))}

    # Compare and find out directories present in source but not in target
    dirs_to_move = source_subdirs - target_subdirs

    # Move directories
    for dir_name in dirs_to_move:
        src = os.path.join(source_dir, dir_name)
        dst = os.path.join(new_dir, dir_name)
        shutil.move(src, dst)
        print(f"Moved {dir_name} to {new_dir}")

# Example usage:
# compare_and_move("/path/to/source_dir", "/path/to/target_dir", "/path/to/new_dir")


#### end file manip ####

        
### get dicom attribute ###

def get_contrast_value(filename, output_dir):
    
    ds = pydicom.dcmread(filename,force=True)
    contrast = getattr(ds, "ContrastBolusAgent", None)
    sd = getattr(ds, "SeriesDescription", "NS").upper()
    pn = getattr(ds, "ProtocolName", "PN").upper()
    contrast_route = getattr(ds, "ContrastBolusRoute", "PN").upper()
    if contrast != "" and contrast != "LO" and contrast is not None:
        return "C"

    
    sd_split = split_string(sd)  
    if "KM" in sd_split or ("POST" in sd_split and "CONTRAST" in sd_split) or "CECT" in sd_split or "GAD" in sd_split or "GD" in sd_split:
        return "C"
    pn_split = split_string(pn)
    if "KM" in pn_split or ("POST" in pn_split and "CONTRAST" in pn_split) or "CECT" in pn_split or "GAD" in pn_split or "GD" in sd_split:
        return "C"   
    if contrast_route == "" or contrast_route == "LO" or contrast == "" or contrast_route == "LO":
        name = os.path.join(os.path.dirname(filename),sd)
        write_to_csv('maybe_contrast.csv', [name, ""], output_dir)
    return ""

def get_study_date(ds, root_dir, filename, output_dir, studyDate_files=None):
    studyDate = getattr(ds, "StudyDate", "")

    if studyDate == "":
        relative_path = os.path.relpath(os.path.dirname(filename), root_dir)
        directories = relative_path.split(os.path.sep)

        if len(directories) >= 2 and directories[1].isdigit():
            studyDate = directories[1]
        else:
            studyDate = getattr(ds, "SeriesDate", "")
            if studyDate == "":
                studyDate = getattr(ds, "AcquisitionDate", "")
                if studyDate == "":
                    studyDate = getattr(ds, "StructureSetDate", "")
                    if studyDate == "":
                        studyDate = "NS"
                        write_to_csv('no_studyDate.csv', [os.path.dirname(filename), ""], output_dir)

    return studyDate

def get_study_date2(ds):
    studyDate = getattr(ds, "StudyDate", "")

    if studyDate == "":
        studyDate = getattr(ds, "SeriesDate", "")
        if studyDate == "":
            studyDate = getattr(ds, "AcquisitionDate", "")
            if studyDate == "":
                studyDate = getattr(ds, "StructureSetDate", "")
                if studyDate == "":
                    studyDate = "NS"

    return studyDate


def find_petct_images(dicom_dirs):
    petct_dirs = []

    for dicom_dir in dicom_dirs:
        try:
            dicom_file = os.listdir(dicom_dir)[0]  # grab the first file
            dicom_file_path = os.path.join(dicom_dir, dicom_file)
            ds = pydicom.dcmread(dicom_file_path)
            if ds.Modality == "CT" or ds.Modality =="PT":
                if 'PETCT' in str(ds).upper() or 'PET CT' in str(ds).upper() or 'PET/CT' in str(ds).upper():  # convert metadata to string to search
                    petct_dirs.append(dicom_dir)
            if ds.Modality == "CT":
                if 'PET' in str(ds).upper():
                    petct_dirs.append(dicom_dir)
        except Exception as e:
            print(f"Could not read {dicom_dir}: {e}")

    return petct_dirs

def find_petmr_images(dicom_dirs):
    petmr_dirs = []
    pet_key = ["PT","PET","FAPI","FET"]
    for dicom_dir in dicom_dirs:
        try:
            dicom_file = os.listdir(dicom_dir)[0]  # grab the first file
            dicom_file_path = os.path.join(dicom_dir, dicom_file)
            ds = pydicom.dcmread(dicom_file_path)
            if ds.Modality == "MR" or ds.Modality =="PT":
                if 'PETMR' in str(ds).upper() or 'PET MR' in str(ds).upper() or 'PET/MR' in str(ds).upper():   # convert metadata to string to search
                    petmr_dirs.append(dicom_dir)
            if ds.Modality == "MR":
            
                if 'PET' in str(ds) or 'FET' in str(ds):   # convert metadata to string to search
                    petmr_dirs.append(dicom_dir)
     
            if ds.Modality == "PT":
                if 'MRT' in str(ds) or 'MR' in str(ds):   # convert metadata to string to search
                    petmr_dirs.append(dicom_dir)            
                
        except Exception as e:
            print(f"Could not read {dicom_dir}: {e}")

    return petmr_dirs


# def get_clean_words(text):
#     """
#     Helper function to clean and split the words from the input text.
#     """
#     return split_string(strip_non_ascii(re.sub(r'[^\w]', ' ', text.upper())))
def get_clean_words(text):
    """
    Helper function to clean and split the words from the input text. 
    Preserves the periods between numbers.
    """
    cleaned_text = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', text.upper())
    cleaned_text = re.sub(r'[^\w.]', ' ', cleaned_text)
    return split_string(cleaned_text)

def check_body_part(words, valid_parts):
    """
    Helper function to check the body part based on the words.
    """
    class_label = "NS"
    for part, valid_words in valid_parts.items():
        if any(word in valid_words for word in words):
            if class_label == "NS":
                class_label = part
            else:
                class_label = class_label + '-' +part
    if "WB" in class_label:
        class_label = "WB"
    return class_label
    
def remove_duplicates(label):
    parts = label.split('-')
    unique_parts = set(parts)
    return '-'.join(sorted(unique_parts))

def reorder_labels(label):
    order = ["HNC", "LUNG", "ANP"]
    # Split the label into individual parts
    parts = label.split('-')

    # Function to get the sort key, unknown labels go to the end
    def sort_key(x):
        return (order.index(x) if x in order else len(order), x)

    # Sort the parts based on the predefined order
    sorted_parts = sorted(parts, key=sort_key)
    if "WB" in sorted_parts:
        sorted_parts = "WB"
    

    # Join the sorted parts back into a string
    return '-'.join(sorted_parts) 

def get_bodypart(ds):
    """
    Extract the body part scanned from a DICOM directory.
    """    
    valid_parts = {
        "HNC": {"hirn","KA_0","SchÃ¤de",
                "Sch?del","CTSCHDEL","CAROTID","HALS","Schaedel","Sch?del","SCHADEL","HEADNECK", 
                "Kopf", "OPTICCANAL","NEUROCRANIUM", "BRAIN","CTSCHDELNATIV", "HEAD", "HEADNECK", "SKULL",
                "NECK","MRHJARNA","NEURO","Hal","schwannoma","SCHÄDEL","Schï¿½del","Schdel"},
        "ANP": {"ABD","Abdomen","Becken","ABDOMENPELVIS","SACRUM","body","PANCREAS","COLON",
                "VASCRENAL","RECTUM","Abdomenbereich","PROSTATE","LIVER","KIDNEY","HIP","PELVIS","GENERALPELVIS"},
        "LUNG": {"THO","Mammographie","CHEST","THORAX","CTTHORAXMITKM","BREAST","HEART","LUNG","LUNGE"},
        "WB": {"WHOLEBODY","WHOLE","WB"},
        "WS": {"gesamte WirbelsÃ¤ul","Wirbelsäule",
                    "Wirbelsï¿½ule","WirbelsÃ¤ule", "spine","myelography","WHOLESPINE"},
        "TS": {"BWS"},
        "LS": {"LWS"},
        "CS": {"HWS"},
        "ARM": {"ARM","ELBOW","HAND"},
        "SHLDR": {"SHOULDER","Schulter","Clavicula"},
        "KNEE": {"Kniegelenk","knee"},
        "LEG":{"LEG","BEINE","FOOT","BEIN"}
    }    
    valid_parts = {k: {word for phrase in v for word in get_clean_words(phrase)} for k, v in valid_parts.items()}
    
    # valid_parts = {k: {get_clean_words(word) for word in v} for k, v in valid_parts.items()}
    class_label = None
    
    value = getattr(ds, "SeriesDescription",None)
    if value:
        words = get_clean_words(value)
        if words:
            label = check_body_part(words, valid_parts)
            if label != "NS" and label:
                print(value)
                return label
    fields = ['BodyPartExamined', 'StudyDescription','RequestedProcedureDescription']
    
    for field in fields:
        try:
            value = getattr(ds, field)
            print(value)
            if not value:
                #continue
                print("mising")
            words = get_clean_words(value)
 
            label = check_body_part(words, valid_parts)
       
           
            if label != "NS":
                if class_label and label not in class_label:
                    class_label = class_label+ '-' + label
                if not class_label:
                    class_label = label
                
        except AttributeError:
       
            continue

    if class_label:
        
        return reorder_labels(remove_duplicates(class_label))

    fields = [ 'RequestedProcedureDescription', 
              'ProcedureCodeSequence[0].CodeMeaning', 'SeriesDescription','ViewDirection',
              'PatientRestDirection','AdmittingDiagnosesDescription','TransmitCoilName']
    
    for field in fields:
        try:
            if field.startswith('ProcedureCodeSequence') and hasattr(ds, 'ProcedureCodeSequence') and ds.ProcedureCodeSequence:
                words = get_clean_words(str(ds.ProcedureCodeSequence[0].CodeMeaning))
            else:
                value = getattr(ds, field)
                if not value:
                    continue
                words = get_clean_words(value)
            if "BODY" in words and "WHOLE" in words:
                return "WB"
            label = check_body_part(words, valid_parts)
            if label != "NS":
                return reorder_labels(remove_duplicates(label))
        except AttributeError:
            continue
    for attr in ds.dir():
        if hasattr(ds, attr):
            value = getattr(ds, attr)
            if value and (isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value))):
                words = get_clean_words(str(value))
                label = check_body_part(words, valid_parts)
                if label != "NS" and label:
                    ee = words
                    at = attr
            
                    return reorder_labels(remove_duplicates(label))
    return "NS"


from Levenshtein import distance


def get_image_orientation(dcm):
    """Retrieve the Image Orientation attribute from a DICOM object."""
    closest_attr = None
    # Directly try to get the ImageOrientationPatient attribute
    image_ori = getattr(dcm, "ImageOrientationPatient", None)
    if not image_ori:
        # If not found, try the retired attribute name
        image_ori = getattr(dcm, "RETIRED_ImageOrientation", None)
    
    if not image_ori:
        # If still not found, find the closest match
        target = "ImageOrientation"
        
        # Get all attributes of the DICOM object
        all_attributes = [attr for attr in dir(dcm) if not callable(getattr(dcm, attr,None)) and not attr.startswith("__")]
        if all_attributes:
            # Find the attribute with the smallest Levenshtein distance to "ImageOrientation"
            closest_attr = min(all_attributes, key=lambda x: distance(target, x))
            
            # Fetch the value of the closest attribute
            potential_image_ori = getattr(dcm, closest_attr, None)
            
            # Check if the potential attribute's value is a list of length 6 and contains numbers
            if isinstance(potential_image_ori, (list, tuple)) and len(potential_image_ori) == 6 and all(isinstance(i, (int, float)) for i in potential_image_ori):
                image_ori = potential_image_ori
                print(f"Using the closest attribute to ImageOrientation: {closest_attr}")
    
    return image_ori, closest_attr


def get_scan_orientation(dcm_file):
    # Read the DICOM file to extract the orientation metadata
    if isinstance(dcm_file, list) and len(dcm_file) > 1:
        dcm = dcm_file[1]
    else:
        dcm = pydicom.dcmread(dcm_file, force=True)

    orientation_name = "NA"

    try:
        aquisition = dcm.MRAcquisitionType
        if aquisition == "2D":
            aquisition = ""
    except:
        aquisition = ""

    # Fallback to image orientation analysis if no keyword is found
    image_ori, att = get_image_orientation(dcm)
    # if att:
    #     print("found orientation attribute " + dcm_file)

    if image_ori:
        image_y = np.array([image_ori[0], image_ori[1], image_ori[2]])
        image_x = np.array([image_ori[3], image_ori[4], image_ori[5]])
        image_z = np.cross(image_x, image_y)
        abs_image_z = abs(image_z)
        orientation = list(abs_image_z).index(max(abs_image_z))
        # Map orientation to its corresponding name
        orientation_name = ""
        if orientation == 0:
            orientation_name = "SAG"
        elif orientation == 1:
            orientation_name = "COR"
        elif orientation == 2:
            orientation_name = "TRA"
        elif orientation == 3:
            orientation_name = "3D"
    ### determine through sd or pt        
    axial = ["TRA", "AXIAL", "TRANSVERSAL", "AX"]
    coronal = ["COR", "CORONAL"]
    sagittal = ["SAG", "SAGITTAL"]
    orientation_keywords = axial + coronal + sagittal
    orientation_keyword = ""           
    if orientation_name == "NA":
        
        sd = getattr(dcm, "SeriesDescription", "NS").upper()
        sd_words = split_string(sd)
        # Initialize variables
        aquisition = ""

        # Check for orientation keywords in SeriesDescription

        for word in sd_words:
            if word in orientation_keywords:
                orientation_keyword = word

        # Determine orientation based on the last keyword found
        if orientation_keyword in axial:
            orientation_name = "TRA"
        elif orientation_keyword in coronal:
            orientation_name = "COR"
        elif orientation_keyword in sagittal:
            orientation_name = "SAG"
    if orientation_name == "NA":
        
        sd = getattr(dcm, "ProtocolName", "NS").upper()
        sd_words = split_string(sd)
        # Initialize variables
        aquisition = ""
       
        orientation_keyword = ""
        # Check for orientation keywords in SeriesDescription
    
        for word in sd_words:
            if word in orientation_keywords:
                orientation_keyword = word
    
        # Determine orientation based on the last keyword found
        if orientation_keyword in axial:
            orientation_name = "TRA"
        elif orientation_keyword in coronal:
            orientation_name = "COR"
        elif orientation_keyword in sagittal:
            orientation_name = "SAG"
        
    return aquisition + orientation_name

### end get dicom attribute ###

####### initial curation scripts ####

def copy_files_for_key(args):
    key, files, output_dir, subject_id_position, key_index = args
    for file in files:
        copy_file((key, file, output_dir, subject_id_position, key_index))


def rearrange_files(root_dir, output_dir = None):
    
    subject_id_position = len(root_dir.split('/'))
    print('Starting rearrangement of files...')

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(root_dir), strip_non_ascii(re.sub(r'[^\w]', '_', os.path.basename(root_dir))) + "_PsCurated")
    os.makedirs(output_dir, exist_ok=True)

    print('Gathering DICOM files...')
    dicom_files = []
    for path, subdirs, files in tqdm(os.walk(root_dir), desc="Gathering DICOM files", dynamic_ncols=True):        
        for f in files:
            if os.path.getsize(os.path.join(path,f)) != 0 and ".DS_Store.dcm" not in f and '.dcm' in f:
                dicom_files.append(os.path.join(path, f))
    
   
    print('Processing DICOM files...')
    scans = defaultdict(list)
    # Prepare the arguments for process_file
    file_args = [(file, output_dir, root_dir) for file in dicom_files]
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        for key, filename in tqdm(executor.map(process_file, file_args),
                                  total=len(dicom_files), desc="Processing DICOM files", dynamic_ncols=True):            
            if key is not None:
                scans[key].append(filename)

    print('Starting to copy files...')
    scans_keys = list(scans.keys())
    with Pool(8) as pool:
        args = [(key, scans[key], output_dir, subject_id_position, str(scans_keys.index(key))) for key in scans_keys]
        for _ in tqdm(pool.imap(copy_files_for_key, args), total=len(args), desc="Copying files", dynamic_ncols=True):
            pass
    print('Copying of files completed.')

    print('Rearrangement of files completed.')

    return output_dir


def check_loc(filename):
    valid_loc = ["Patient Aligned MPR",
                 "SCOUT","LOCALIZER","PROJECTION IMAG", "PROJECTION IMAGE",
                 "LOKALIZER", "POSDISP", "SURVEY","LOC","FASTVIEW","LOK","SMARTBRAIN"]
    ds = pydicom.dcmread(filename,force=True)
    protocolName = getattr(ds, "ProtocolName", "PN").upper()
    sd = getattr(ds, "SeriesDescription", "NS").upper()
    if any(word in valid_loc for word in getattr(ds, "ImageType", "NS")) or any(word in valid_loc for word in split_string(protocolName))  or any(word in valid_loc for word in split_string(sd)):
        return True
    else:
        return False

def process_file(args):
    filename, output_dir, root_dir = args
    ImageTypes = []
    try:
        ds = pydicom.dcmread(filename,force=True)
    except:
        print('{} could not be read, dicom file may be corrupted'.format(filename))
        return None, filename
    
    valid_loc = ["SCOUT","LOCALIZER", "PROJECTION IMAGE","Lokalizer", "POSDISP"]
    patientName = str(getattr(ds, "PatientName", "PN")).upper().replace('_','')
    protocolName = getattr(ds, "ProtocolName", "PN").upper().replace('_','')
    if protocolName == "":
        protocolName = "NS"

    sd = getattr(ds, "SeriesDescription", "NS").upper()
    seriesDescription = getattr(ds, "SeriesDescription", "NS").upper().replace('_','')
    if seriesDescription == "":
        seriesDescription = "NS"
    if any(word in valid_loc for word in getattr(ds, "ImageType", "NS")) or any(word in valid_loc for word in split_string(protocolName))  or any(word in valid_loc for word in split_string(sd)):
        print('{} scout scans are ignored'.format(filename))
        return None, filename
    
    if seriesDescription == "":
        seriesDescription = "NS"
    
    body_part = get_bodypart(ds) # You would need to define the get_bodypart function
    modality = get_modality(ds)
    # modality = getattr(ds, "Modality", "Empty").upper().replace('_','')
    # if modality == "OT":
    #     bp = getattr(ds, "BodyPartExamined", "NS").replace('_','')
    #     if "MR" in bp :
    #         modality = "MR"
    #     elif "CT" in bp:
    #         modality = "CT"    
    #     else:
    #         if "CT " in str(ds):
    #             modality = "CT"
    #         elif "MR " in str(ds):
    #             modality = "MR"

        
    imageTypes = str(getattr(ds, "imageTypes", "NS"))
    if imageTypes != "NS":
        imageTypes_str = ''.join(imageTypes).replace('_','')
    else:
        imageTypes_str = ""
        
    scanOptions = str(getattr(ds, "ScanOptions", "NS"))
    if scanOptions != "NS" and modality == "MR":
        scanOptions_str = ''.join(scanOptions).replace('_','')
    else:
        scanOptions_str = ""
  
    angio = getattr(ds, "AngioFlag", "NS")
    
    sliceLocation = str(getattr(ds, "SliceLocation", "NS")).replace('_','')
    studyInstance = getattr(ds, "StudyInstanceUID", "NS").replace('_','')
    seriesInstance = getattr(ds, "SeriesInstanceUID", "NS").replace('_','')
    studyDate = get_study_date(ds, root_dir, filename, output_dir,studyDate_files=None)
    receiveCoilName = getattr(ds, "ReceiveCoilName", "NS").replace('_','')
    rt_modalities = ["RTPLAN","RTSTRUCT","RTDOSE","RTIMAGE"]
    # if not hasattr(ds, "PixelData") and modality not in rt_modalities:
    #     modality = "SR"
    #echoTime = str(getattr(ds, "EchoTime", "Empty"))
    contrast = get_contrast_value(filename, output_dir)
    if contrast == "C" and modality != "PR":
        modality = "C"+modality
    try:    
        orientation = get_scan_orientation(filename)
    except:
        orientation = "NA"
      
    if modality == "PR" or modality == "SR":
        orientation, body_part = "NA","NA"

    if "CBCT" in seriesDescription or "CONEBEAM" in seriesDescription:
        modality = "CBCT"
    if "FS" in scanOptions or scanOptions == "FS":
        modality = modality + "-FS"
    if angio == "Y":
        modality = "MRA"
    ###### we need a function that checks the MRA folders and labels the MRA-PRE, MRA-POST and MRA
    sequence_type = determine_sequence_type(filename)
    if sequence_type != "NS":
        modality = modality + "-" + sequence_type

    key= handle_none(patientName)+'_'+handle_none(studyDate) + '_' + handle_none(orientation) + \
        '_' + handle_none(body_part) + '_' + handle_none(modality) + '_' + handle_none(seriesDescription) + \
            '_' + handle_none(protocolName)+'_'+ handle_none(seriesInstance) + '_' + handle_none(studyInstance) + \
                '_'+ handle_none(contrast)+'_'+handle_none(receiveCoilName)+'_'+ imageTypes_str+scanOptions_str 
    #key= studyDate + '_' + orientation+ '_' + body_part + '_'+ modality + '_' + seriesDescription +'_' + protocolName+'_'+ seriesInstance + '_' + studyInstance +'_'+ contrast+'_'+receiveCoilName
    key=strip_non_ascii(re.sub(r'[^\w]', '', key))
    key=key.replace('_','-')

    return key, filename+'_sloc'+sliceLocation

def get_modality(ds):
    """
    Determines the modality of a DICOM dataset.
    Args:
    ds: A DICOM dataset.

    Returns:
    A string representing the determined modality (e.g., "MR", "CT", or original modality).
    """
    modality = getattr(ds, "Modality", "Empty").upper().replace('_', '')
    if modality == "OT":
        bp = getattr(ds, "BodyPartExamined", "NS").replace('_', '')
        if "MR" in bp:
            modality = "MR"
        elif "CT" in bp:
            modality = "CT"
        else:
            # Check within the string representation of the dataset
            if "CT" in split_string(str(ds)):
                modality = "CT"
            elif "MR" in split_string(remove_special_chars_and_numbers(str(ds),False)) or "FremdMRT" in split_string(remove_special_chars_and_numbers(str(ds),False)) :
                modality = "MR"
    return modality

def determine_sequence_type(dicom_file_path):
    """
    Determines if a DICOM file's imaging sequence is Gradient Echo (GE), Spin Echo (SE), or Inversion Recovery (IR)
    based on the ScanningSequence, ImageType, SequenceName, and SeriesDescription attributes.
    
    Parameters:
    - dicom_file_path: Full path to the DICOM file to be checked.
    
    Returns:
    - "GE" if Gradient Echo sequence indicators are found.
    - "SE" if Spin Echo sequence indicators are found.
    - "IR" if Inversion Recovery sequence indicators are found.
    - "NS" if the attribute is not present or no specific indicators are identified.
    """
    try:
        # Load the DICOM file
        dicom_file = pydicom.dcmread(dicom_file_path)
        
        # Keywords for GRE, SE, and IR, converted to lowercase for case-insensitive comparison
        gre_keywords = ['mpr', 'fl3d', 'fl2d', 'gre']
        se_keywords = ['tse', 'se', 'fse', 'flair']
        ir_keywords = ['stir', 'flair', 'tir', 'inversion recovery', 'ir']
        
        # Get attributes with getattr, returning 'NS' if not found
        scanning_sequence = getattr(dicom_file, "ScanningSequence", "NS")
        image_type = getattr(dicom_file, "ImageType", "NS")
        sequence_name = getattr(dicom_file, "SequenceName", "NS").lower()  # Convert to lowercase
        series_description = getattr(dicom_file, "SeriesDescription", "NS").lower()  # Convert to lowercase
        print(scanning_sequence)
        # Check 'ImageType' attribute for GRE, SE, and IR keywords
        if image_type != "NS":
            image_type = [x.lower() for x in image_type]  # Convert to lowercase for comparison
            if any(keyword in image_type for keyword in gre_keywords):
                return "GE"
            if any(keyword in image_type for keyword in se_keywords):
                return "SE"
            if any(keyword in image_type for keyword in ir_keywords):
                return "IR"
        
        # Check 'SequenceName' and 'SeriesDescription' attribute for GRE, SE, and IR keywords
        for attribute in [sequence_name, series_description]:
            if attribute != "NS":
                if any(keyword in attribute for keyword in gre_keywords):
                    return "GE"
                if any(keyword in attribute for keyword in se_keywords):
                    return "SE"
                if any(keyword in attribute for keyword in ir_keywords):
                    return "IR"
        
        # Check 'ScanningSequence' for 'GR' (indicative of GE), 'SE', and 'IR'
        if scanning_sequence != "NS":
            if 'GR' in scanning_sequence:
                return "GE"
            if 'SE' in scanning_sequence:
                return "SE"
            if 'IR' in scanning_sequence:
                return "IR"
            return scanning_sequence
        
        # If no specific indicators found
        return "NS"
                
    except Exception as e:
        print(f"Error processing file {dicom_file_path}: {e}")
        return "Error"


def check_flair_type(dcm_file):
    try:
        # Read the DICOM file
        dicom_data = pydicom.dcmread(dcm_file)
        
        # Get the Repetition Time (TR) and Echo Time (TE)
        tr = dicom_data.get('RepetitionTime', None)  # In milliseconds
        te = dicom_data.get('EchoTime', None)  
        print(tr)
        print(te)# In milliseconds
        
        # Get the SeriesDescription to check if it's T1 or T2 weighted
        series_description = dicom_data.get('SeriesDescription', '').upper()
        
        # Check if 'T1' or 'T2' is in the SeriesDescription
        if 'T1' in series_description:
            return 'T1FL'
        elif 'T2' in series_description:
            return 'T2FL'
        else:
      
            # Determine if it's T1-weighted or T2-weighted FLAIR based on TR and TE values if not specified
            if tr is not None and te is not None:
                if te < 40:
                    return 'T1FL'
                else:
                    return 'T2FL'
        
        # If the sequence type could not be determined
        return 'Unknown FLAIR type'

    except Exception as e:
        print(f"An error occurred while processing {dcm_file}: {e}")
        return 'Error processing file'



def create_patient_dict(file_paths, id_pos=6):
    # Create empty dictionaries to store patient IDs and time points
    patient_ids = {}
    time_points = {}
    patient_dict = {}

    # Loop over the list of file paths
    for file_path in file_paths:
        # Split the file path into its components
        path_parts = file_path.split('/')
        
        # Extract the patient ID and time point from the path components
        patient_id = path_parts[id_pos]
        time_point = path_parts[id_pos+1]
        
        # Encode the patient ID as a number starting with 1
        if patient_id not in patient_ids:
            patient_ids[patient_id] = len(patient_ids) + 1
            time_points = {}
        patient_id_num = patient_ids[patient_id]
        
        # Encode the time point as a number starting with 1 for this patient
        if time_point not in time_points:
            time_points[time_point] = len(time_points) + 1
        time_point_num = time_points[time_point]
        
        # Add the patient ID and time point numbers to the patient_dict dictionary
        if file_path not in patient_dict:
            patient_dict[file_path] = (patient_id_num, time_point_num)

    return patient_dict



def convert_and_label_4d_png_txt_pl(output_dir,modality_considered = modality_considered, convert_to =".nii.gz"):
    print('Starting conversion and labeling...')
    
    # ignore_list = ['CXA','XA','CR','UNCONVERTABLE','RTDOSE', 'RTSTRUCT', 
    #                'RTPLAN', 'RTRECORD', 'RTIMAGE','SR','PR','REG','KO','US','NM','ECG','RF']
    # dcm_folders = [x for x in glob.glob(output_dir + '/*/*/*/*/*/*') 
    #            if os.path.isdir(x) 
    #            and x.split('/')[-2] not in ignore_list
    #            and not os.path.isfile(os.path.join(os.path.dirname(x), os.path.basename(x) + '.nii.gz'))]

    dcm_folders = [x for x in glob.glob(output_dir + '/*/*/*/*/*/*')
                   if 'RTSTRUCT' not in x and os.path.isdir(x) and 
                   any(word in x.split('/')[-2] for word in modality_considered) and 
                   not '.nii.gz' in x ] 
        
    total_folders = len(dcm_folders)
    print('Processing DICOM folders...')
    
    folders_with_args = [(dicom, output_dir, convert_to) for dicom in dcm_folders]
    # with Pool(cpu_count()) as p:
    #     p.map(process_folder, dcm_folders)
        
    # Use imap_unordered and tqdm together to create a progress bar
    with Pool(8) as p:
        pbar = tqdm(total=total_folders)  # Initialize the progress bar
        for _ in p.imap_unordered(process_folder_helper, folders_with_args):
            pbar.update()  # Update the progress bar each time a result is ready
    print('Processed {} folders...'.format(total_folders))        

def process_folder_helper(args):
    return process_folder(*args)

# def rename_dicom_files(dcm_files):
#     # Create a dictionary to store the used filenames within each folder
#     used_filenames = {}
    
#     # Initialize a counter
#     counter = 1
    
#     # Iterate through the sorted files and rename them with their folder structure preserved
#     for old_filepath in dcm_files:
#         # Split the file path into directory and filename
#         directory, filename = os.path.split(old_filepath)
        
#         # Get the file extension
#         file_extension = os.path.splitext(filename)[1]
        
#         # Create the new filename with the counter
#         new_filename = f"{counter:04d}{file_extension}"  # Use 4-digit numbers, e.g., 0001, 0002, ...
        
#         # Check if the new filename already exists in the folder; if so, increment the counter
#         while os.path.join(directory, new_filename) in used_filenames.get(directory, []):
#             counter += 1
#             new_filename = f"{counter:04d}{file_extension}"
        
#         # Add the new filename to the list of used filenames in the folder
#         used_filenames.setdefault(directory, []).append(os.path.join(directory, new_filename))
        
#         # Build the new file path
#         new_filepath = os.path.join(directory, new_filename)
        
#         # Rename the file
#         os.rename(old_filepath, new_filepath)
        
#         # Increment the counter
#         counter += 1


def process_folder(dicom, output_dir, convert_to=".nii.gz"):

    #sprint(dicom)
    cont = True
    label = None
    dcm_files = glob.glob(dicom+'/*')
    # Perfussion check
    if len(dcm_files) > 300:
        ds = pydicom.dcmread(dcm_files[0],force=True)
        if "MOSAIC" in ds.ImageType:
            move_folders_to_new_dir([dicom], "MOSAIC")
            return "MOSAIC"
        sd = getattr(ds, "SeriesDescription", "").upper()
        pt = getattr(ds, "ProtocolName", "").upper()
        md = getattr(ds, "Modality", "").upper()
        if check_per(dicom,ds,sd,pt):
            if md =="CT":
                label = "4DPERCT"   
            else:
                label = "4DPER"
            
        elif check_dwi(dicom,ds,sd,pt):
            label = "4DDWI"
        elif check_dti(dicom,ds,sd,pt):
            label = "4DDTI"
        else:
            att = get_varying_dicom_attributes(dicom)
            if att:
                label = "4D"+ dicom.split("/")[-2]
        if label:
            move_folders_to_new_dir([dicom], label)
        return label
    try:
        dcm_files = order_dicom_files(dcm_files)
    except:
        dcm_files = dcm_files

    # Convert images
    if convert_to == ".nii.gz":
        conv_image, cont = convert_image(dicom)
        if cont:
            label = load_and_check_image(dicom,  output_dir)
        else:
            label = "UNCONVERTABLE"
            move_folders_to_new_dir([dicom], label)

    # Clear temporary variables at the end of loop
    gc.collect()
    return label

def check_conversion_issues(dicom_path, nifti_path):
    """
    Checks for potential issues between a DICOM file and a converted NIfTI file.

    :param dicom_path: Path to the original DICOM file.
    :param nifti_path: Path to the converted NIfTI file.
    :return: True if there is a problem, False otherwise.
    """
    # Load the DICOM file
    dicom_img = pydicom.dcmread(glob.glob(dicom_path+'/*')[0])
    
    # Extract relevant DICOM attributes
    dicom_slice_thickness = float(dicom_img.SliceThickness)
    dicom_pixel_spacing = tuple(map(float, dicom_img.PixelSpacing))

    # Load the NIfTI file
    nifti_img = nib.load(nifti_path)
    nifti_data = nifti_img.get_fdata()
    nifti_header = nifti_img.header

    # Check the image dimensions
    if nifti_data.shape[:2] != dicom_img.pixel_array.shape:
        print("Issue: Mismatched image dimensions.")
        return True

    # Check for NaN or infinite values
    if np.isnan(nifti_data).any() or np.isinf(nifti_data).any():
        print("Issue: The NIfTI image contains NaN or infinite values.")
        return True

    # Check slice thickness and pixel spacing (voxel sizes)
    nifti_voxel_sizes = nifti_header.get_zooms()[:2]
    if not np.allclose(nifti_voxel_sizes, dicom_pixel_spacing, atol=1e-3):
        print("Issue: Mismatched pixel spacing.")
        return True
    if not np.isclose(nifti_voxel_sizes[2], dicom_slice_thickness, atol=1e-3):
        print("Issue: Mismatched slice thickness.")
        return True

    # No issues found
    return False

def get_compressed_files(dicom_directory):
    compressed_files = []
    for file_path in glob.glob(dicom_directory + '/*.dcm'):
        if check_transfer_syntax_if_compress(file_path):
            compressed_files.append(file_path)
    return compressed_files

def convert_image(dicom,convert_to=".nii.gz"):
    

    output_dir = '/'.join(dicom.split('/')[:-6])
    compressed_files = get_compressed_files(dicom)
    if compressed_files:
        decompress_CT(compressed_files)
  
    conv_image = DicomConverters(dicom).dcm2niix_converter()
    if conv_image:
        try:
            dicom2nifti.dicom_series_to_nifti(dicom,dicom+convert_to,reorient_nifti=False)
            failed = False
        except:
            failed = True
        if failed:
            decompress_CT(dicom)
            conv_image = DicomConverters(dicom).dcm2niix_converter()
            if not conv_image:
                write_to_csv('decompress_convert.csv', [dicom, ""], output_dir)
            else:
                conv_image = convert_dicom_sitk(dicom)
                if not conv_image:
                    write_to_csv('sitk_convert.csv', [dicom, ""], output_dir)
                else:
                    write_to_csv('failed_convert.csv', [dicom, ""], output_dir)
                    return conv_image, False
    return conv_image, True

def load_and_check_image(dicom, output_dir):
    
    label = None
    try:
        img_data = nib.load(dicom+'.nii.gz').get_fdata() 
    except Exception as e:
        if "ufunc" in str(e):
            label = "UNREADABLE"
            move_folders_to_new_dir([dicom], label)
            write_to_csv('failed_read.csv', [dicom, label], output_dir)
            return label
        else:
            print("ERRORRRRRR"+dicom)
    if is_rgb(dicom+'.nii.gz')[0]:
        label,orientation_relabel = label_rgb(dicom+'.nii.gz')
        if label:
            if orientation_relabel:
                dicom = modify_and_move(dicom, -4, orientation_relabel, replace=True)
            move_folders_to_new_dir([dicom], label)
            move_folders_to_new_dir([dicom+'.nii.gz'], label)
            return label
        
        
    ####### change to consider the dicom not ht nifty    
    if len(img_data.shape) == 4:
        try:
            dicom2nifti.dicom_series_to_nifti(dicom, dicom+'.nii.gz',reorient_nifti=False)
            img_data2 = nib.load(dicom+'.nii.gz').get_fdata() 
        except:
            img_data2 = img_data

        if len(img_data2.shape) == 4: 
            label = process_4d(dicom, label, output_dir,img_data.shape)

    if label and label != "PDT2":
        print(dicom)
        move_folders_to_new_dir([dicom], label)
        move_folders_to_new_dir([dicom+'.nii.gz'], label)
    return label

def check_for_mra(dcm_file):
    # Load the DICOM file
    ds = pydicom.dcmread(dcm_file)
    
    # Check if angioFlag is on
    if hasattr(ds, 'angioFlag') and ds.angioFlag:  # Adjust 'angioFlag' as necessary
        return "Angio Flag is on"
    
    # Define keywords to search for in the SeriesDescription
    keywords = [
        "carotis", "twist", "ToF", "Phase Contrast", "Inhance Velocity Phase Contrast",
        "Phase Shift", "TWIST", "TRICKS-XV", "Keyhole", "4D-TRAK", "DRKS", "TRAQ", "NATIVE-SPACE",
        "Inhance Deltaflow", "TRANCE", "FBI", "CIA", "CARE", "Bolus", "Smart Prep", "Fluoro Triggered",
        "MRA", "BolusTrak", "Visual Prep", "FLUTE", "Multi-Slab", "MOTSA", "TONE", "Ramped RF",
        "ISCE", "SSP", "MTC", "SORS-STC", "QISS", "NATIVE-TrueFISP", "Inhance Inflow", "IR", "B-TRANCE",
        "Time-SLIP", "VASC"
    ]
    
    # Ensure SeriesDescription is present
    if hasattr(ds, 'SeriesDescription'):
        series_description = ds.SeriesDescription.lower()
        
        # Check if any of the keywords are in the SeriesDescription
        for keyword in keywords:
            if keyword.lower() in series_description:
                return "MRA"
    
    # Return a default value if no conditions are met
    return "Not MRA"





def split_pca(dicom):
    pca_group = []  # List to hold files with PCA in ImageType
    t1_group = []  # List to hold other files
    sequence_type = None
    # Iterate over files in the given folder
    for filename in os.listdir(dicom):
        if filename.endswith(".dcm"):
            file_path = os.path.join(dicom, filename)
            try:
                # Read the DICOM file
                ds = pydicom.dcmread(file_path)
                
                # Check if 'PCA' is in the ImageType attribute
                if 'PCA' in ds.ImageType:
                    pca_group.append(file_path)
                else:
                    if not sequence_type:
                        sequence_type = determine_sequence_type(file_path)
                    t1_group.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Now pca_group and other_group contain the respective file paths
    # You can move/copy them to other directories if needed
    pca_dir = os.path.join(os.path.dirname(os.path.dirname(dicom)), "MRA",os.path.basename(dicom))
    t1_dir = os.path.join(os.path.dirname(os.path.dirname(dicom)), "T1-"+sequence_type
                          ,os.path.basename(dicom))
    os.makedirs(pca_dir, exist_ok=True)
    os.makedirs(t1_dir, exist_ok=True)

    for file in pca_group:
        shutil.move(file, pca_dir)

    for file in t1_group:
        shutil.move(file, t1_dir)
        

    conv_image, cont = convert_image(pca_dir)
    conv_image, cont = convert_image(t1_dir)
    os.rmdir(dicom)
    os.remove(dicom+'.nii.gz')

def process_4d(dicom, label, output_dir, shape):
    ds = pydicom.dcmread(glob.glob(dicom+'/*')[0], force=True)
    sd = getattr(ds, "SeriesDescription", "").upper()
    pt = getattr(ds, "ProtocolName", "").upper()
    modality = getattr(ds, "Modality", "").upper()
    
    #phase MRA also in imagetype
    if "PCA" in sd and  shape[-1]==2:
        split_pca(dicom)
        return
        
        

    if modality == "CT":
        return "4DCT"
    
    if any(substring in modality for substring in ["MR", "CMR", "OT"]):
        label = check_per(dicom, ds, sd, pt) or check_dwi(dicom, ds, sd, pt)
        if label:
            return label

        if "BOLUS" in sd or "BOLUS" in pt:
            return "TESTBOLUS"

        label =  check_vai(dicom,output_dir)
        if not label:
            label = check_split_multiple_echo_times(dicom,output_dir)

        if label:
            write_to_csv('multiple_echotimes.csv', [dicom, label], output_dir)
        else:
            
            if split_dicom_by_scan_options(dicom,output_dir):
                write_to_csv('multiple_scanoptions.csv', [dicom, label], output_dir)
    

    if not label:

        label = "4D" + modality if "4D" not in modality else modality
        
        write_to_csv('scans_4d.csv', [dicom, label, shape], output_dir)
        
    return label

                #varying_attributes = get_varying_dicom_attributes(dicom)
            # if 'ImagePositionPatient' not in varying_attributes and 'AcquisitionTime' in varying_attributes:
            #     return 'FMRI'
        # mainly for dixon
    

def split_dicom_by_attribute(source_dir, attribute_name):
    dest_dir = os.path.dirname(source_dir)
    for filename in os.listdir(source_dir):
        if filename.endswith('.dcm'):
            full_path = os.path.join(source_dir, filename)
            dicom_file = pydicom.dcmread(full_path)

            # Extract the specified attribute
            attribute_value = getattr(dicom_file, attribute_name, None)

            # If attribute is not found, skip
            if attribute_value is None:
                print(f"Attribute {attribute_name} not found in {filename}. Skipping.")
                continue

            # If attribute is a list or MultiValue object, convert to a string
            if isinstance(attribute_value, (list, pydicom.multival.MultiValue)):
                attribute_value = "_".join([str(value) for value in attribute_value])
            else:
                attribute_value = str(attribute_value)

            # Create a directory for this attribute value if it doesn't exist
            attribute_dir = os.path.join(dest_dir, attribute_value)
            os.makedirs(attribute_dir, exist_ok=True)

            # Copy the file to the corresponding directory
            shutil.copy(full_path, os.path.join(attribute_dir, filename))

    print(f'DICOM files split by {attribute_name}.')
    


def check_vai(dicom,output_dir):
    dicom_files = [os.path.join(dicom, f) for f in os.listdir(dicom) if f.endswith('.dcm')]
    echo_times = {pydicom.dcmread(dcm).EchoTime for dcm in dicom_files}
    vai = []
    new_folders = []  # List to hold new folders

    if len(echo_times) > 1: # There are multiple echo times
        for echo_time in echo_times:
            # Create a directory for each echo time
            new_folder = os.path.join(os.path.dirname(dicom), os.path.basename(dicom)+'_e'+str(echo_time))
            new_folders.append(new_folder)  # Append new folder to list
            os.makedirs(new_folder, exist_ok=True)

            # Copy the DICOM files for each echo time to the new directory
            echo_files = [dcm for dcm in dicom_files if pydicom.dcmread(dcm).EchoTime == echo_time]
            for file in echo_files:
                shutil.copy(file, new_folder)
    
         
            conv_image, cont = convert_image(new_folder)
            if cont:
                # Read the NIfTI file to check its shape
                nifti = nib.load(new_folder+ '.nii.gz')
                if len(nifti.shape) == 4:
                    vai.append(True)
                else:
                    for folder in new_folders:
                        shutil.rmtree(folder)
                        os.remove(folder+'.nii.gz')
                    
                    
                    return None
            else:
                return None
    
        # After checking all echo times, remove new directories
        for folder in new_folders:
            shutil.rmtree(folder)
        
        if all(vai):
            if "VAI" not in pydicom.dcmread(dicom_files[0]).SeriesDescription.upper():
                write_to_csv('maybe_vai.csv', [dicom, "VAI"], output_dir)
            return "VAI"

        
def check_per(dicom,ds,sd,pt):
    keywords = ['PERFUSION', 'PERF','PRESTO_HR', 'PWI']
    if any(keyword in ds.ImageType for keyword in keywords) or any(keyword in sd for keyword in keywords) or any(keyword in pt for keyword in keywords):
        return "4DPER"

def check_dti(dicom,ds,sd,pt):
    keywords = ['DTI', 'DIFFUSION TENSOR','STEAM']
    if any(keyword in ds.ImageType for keyword in keywords) or any(keyword in sd for keyword in keywords) or any(keyword in pt for keyword in keywords):
        return "4DDTI"

def check_dwi(dicom,ds,sd,pt):
    keywords = ['DIFF', 'TRACE', 'DIFFUSION', 'DW','EPI_DIFF','DIFFUS','dDWI']
    for tag in ds.dir(): 
        if 'DIFFUSION' in tag.upper():
            value = getattr(ds, tag)
            if isinstance(value, list) and not np.any(value):
                continue
            if value is not None and value != 0 and value :
                return "4DDWI"
    if "STEAM" not in sd:
        if any(keyword in ds.ImageType for keyword in keywords) or any(keyword in sd for keyword in keywords) or any(keyword in pt for keyword in keywords):
            return "4DDWI"
    ### check from different b values in the sequcen names
    sequence_names = set()  # Set to store unique sequence names
    b_values = set()  # Set to store unique b-values

    for filename in os.listdir(dicom):
        if filename.endswith(".dcm"):
            # Load the DICOM file
            ds = pydicom.dcmread(os.path.join(dicom, filename))
         
            # Save sequence name if it exists
            if "SequenceName" in ds:
                sequence_name = ds.SequenceName
                sequence_names.add(sequence_name)

                # Check for different 'b' values in sequence name
                b_value = re.search('b(\d+)', sequence_name)  # Look for 'b' followed by numbers
                if b_value:
                    b_values.add(int(b_value.group(1)))  # Save b-value as an integer
    
    # Check if there are different 'b' values
    if len(b_values) > 1:
        return "4DDWI"
    return None





def check_dce(folder_path):
    # Get all the subdirectories in the folder_path
    folders = [f for f in glob.glob(os.path.join(folder_path, '*')) if os.path.isdir(f)]
    

    # This dictionary will hold the acquisition times and the corresponding folder names
    acquisition_times = {}
    sub_sequences = {}

    for folder in folders:
        # Find all DICOM files in the subdirectory
        dicom_files = glob.glob(os.path.join(folder, '*.dcm'))
        for dicom_file in dicom_files:
            try:
                # Read the DICOM file
                ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)

                # Check if the DICOM file has necessary attributes
                if all(hasattr(ds, attr) for attr in ['EchoTime', 'RepetitionTime', 'PerformedProcedureStepStartTime']):
                    # Check for 'SUB' in Series Description for sub sequences
                    if 'SUB' in ds.SeriesDescription.upper():
                        sub_sequences[folder] = ds.AcquisitionTime
                    else:
                        acquisition_times[folder] = ds.AcquisitionTime
                    break
            except Exception as e:
                print(f"Failed to read {dicom_file}: {e}")
                continue

    # Sort and rename SUB sequences
    sorted_sub_folders = sorted(sub_sequences.items(), key=lambda item: item[1])
    for i,im in enumerate(sorted_sub_folders):
        new_name = f"{os.path.dirname(im[0])}/{i+1}-{os.path.basename(im[0])}"
        shutil.move(im[0], new_name)
        shutil.move(im[0]+'.nii.gz', new_name+'.nii.gz')
        modify_and_move(new_name, -2, "DCE-SUB" , replace=True)    
 

    # Sort and rename non-SUB sequences
    sorted_folders = sorted(acquisition_times.items(), key=lambda item: item[1])
    for i,im in enumerate(sorted_folders):
        new_name = f"{os.path.dirname(im[0])}/{i+1}-{os.path.basename(im[0])}"
        shutil.move(im[0], new_name)
        shutil.move(im[0]+'.nii.gz', new_name+'.nii.gz')
        modify_and_move(new_name, -2, "DCE" , replace=True)      
    
        
def split_dicom_by_scan_options(dicom_folder,output_dir):
    # Get all DICOM files in the directory
    dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith(".dcm")]

    # Create a set to store the unique ScanOptions across all DICOM files
    unique_scan_options = set()

    # First pass: For each DICOM file, get the ScanOptions attribute and add it to the set
    for dicom_file in dicom_files:
        # Read the DICOM file
        ds = pydicom.dcmread(os.path.join(dicom_folder, dicom_file))
        try:
            scan_options = ds.ScanOptions
        except:
            scan_options = ""

        # Convert MultiValue to a tuple and add it to the set
        if isinstance(scan_options, pydicom.multival.MultiValue):
            unique_scan_options.add(tuple(str(opt) for opt in scan_options))
        else:
            unique_scan_options.add(scan_options)
                
    # If there is only one unique ScanOptions value, return without doing anything
    if len(unique_scan_options) == 1:
        return False

    # Second pass: If there are multiple unique ScanOptions, for each DICOM file
    # create a new directory named after its ScanOptions and move the file there
    # print(unique_scan_options)
    # print(len(unique_scan_options))
    # print(dicom_folder)
    dirs = []
    for dicom_file in dicom_files:
        # Read the DICOM file
        ds = pydicom.dcmread(os.path.join(dicom_folder, dicom_file))

        # Get the ScanOptions attribute
        scan_options = ds.ScanOptions

        # Create the new directory name
        new_parent = os.sep.join(os.path.dirname(dicom_folder).split(os.sep)[:-1] + [os.path.dirname(dicom_folder).split(os.sep)[-1].replace('4D', '')])
        new_dir = os.path.join(new_parent, os.path.basename(dicom_folder)+"_"+scan_options)
        if new_dir not in dirs:
            dirs.append(new_dir)
        # If the directory doesn't exist, create it
        if not os.path.exists(new_dir):
            os.makedirs(new_dir,exist_ok=True)

        # Move the DICOM file to the new directory
        shutil.move(os.path.join(dicom_folder, dicom_file), new_dir)
    for d in dirs:
        convert_image(d)
    # After all files have been processed, remove original folder if it's empty
    if len(os.listdir(dicom_folder)) == 0:
        os.rmdir(dicom_folder)
        os.remove(dicom_folder+'.nii.gz')
    return True
        
def label_rgb(file):
    
    orientation_relabel = None
    
    if '.nrrd' in file:
        dicom = file.split('.nrrd')[0]
    elif '.nii' in file:
        dicom = file.split('.nii')[0]
    else:
        dicom = file
    
    ds = pydicom.dcmread(glob.glob(dicom+'/*')[0])
        
    try:
        seriesDescription = ds.SeriesDescription.upper().replace('_','').replace(" ","")
        if seriesDescription == "":
            seriesDescription = "Empty"
    except:
        seriesDescription = "Empty"
    try:
        imageType = ds.ImageType
    except:
        imageType = "Empty"
        
    col_sequences = ["CBF","CBV","FA","MTT","RELMTT","RELMITT","PBP","TTP"]  # Add your col_sequences here
    label = None  # Define your initial label value

    for col_sequence in col_sequences:
        if col_sequence in imageType and ds.Modality == "MR" :
            if col_sequence == "RELMITT":
                cols = "RELMTT"
            else:
                cols = col_sequence
            label = "COL"+cols
            orientation_relabel = "TRA"
    if label is None:
        for col_sequence in col_sequences:
            if col_sequence in seriesDescription and ds.Modality == "MR":
                if col_sequence == "RELMITT":
                    cols = "RELMTT"
                else:
                    cols = col_sequence
                label = "COL"+cols
    if label is None:
        if "SUMMARY" in seriesDescription:
            label = "SR"
    if label is None:
        petct = find_petct_images([dicom])
        if petct:
            label ="PETCT"
    if label is None:
        petmr = find_petmr_images([dicom])
        if petmr:
            label ="PETMR" 
    
    if label is None:
        label = "PROS" + ds.Modality 

    return label, orientation_relabel    
    
def is_rgb(image_path):
 
    img = sitk.ReadImage(image_path)
    channels = img.GetNumberOfComponentsPerPixel()

    if channels == 3:
        return True, img  # True for 3-channel images

    elif channels == 4:
        pixel_data = sitk.GetArrayFromImage(img)

        # Calculate variance for each channel
        variance = np.var(pixel_data[..., -1])
        if variance == 0:
            return True, img
        else:
            return False, img 
    else:
        return False, img

    


####### end initial curation scripts ####


##### csv preprocessing ###

def filter_paths(csv_file, path_index, match_string):
    df = pd.read_csv(csv_file)
    df = df[~df['Dicom'].apply(lambda x: match_string in os.path.normpath(x).split(os.sep)[path_index])]
    df.to_csv(csv_file, index=False)


def pick_paths(csv_path, position, name):
    df = pd.read_csv(csv_path)
    paths = df['Dicom'].tolist()
    
    filtered_paths = []
    for path in paths:
        path_parts = os.path.normpath(path).split(os.sep)
        if name.lower() == 'all':
            filtered_paths.append(path)
        elif len(path_parts) > abs(position) and name in path_parts[position]:
            filtered_paths.append(path)
    return filtered_paths

def remove_existing_nii_gz_paths(csv_path):
    # Read csv file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Create a mask of boolean values representing whether the .nii.gz file exists for each path
    mask = df['Dicom'].apply(lambda x: not os.path.isfile(x + '.nii.gz'))

    # Keep only the rows in the DataFrame where the .nii.gz file does not exist
    df = df[mask]

    # Write the updated DataFrame back to the csv file
    df.to_csv(csv_path, index=False)

def remove_duplicates_csv(csv_file_path):
    if not os.path.isfile(csv_file_path):
        print(f"{csv_file_path} does not exist.")
        return
    
    # Read the file and store unique rows
    unique_rows = OrderedDict()
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            unique_rows[tuple(row)] = None  # Value doesn't matter, we just care about the keys

    # Now write the unique rows back to the file
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in unique_rows.keys():
            writer.writerow(list(row))
            
    print(f"Duplicates removed from {csv_file_path}")
    
def write_to_csv(file_name, data_row, output_dir):
        # Create the output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    file_exists = os.path.isfile(file_path)
    
    # If the file exists, read its contents to check for the data_row
    if file_exists:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row == data_row:
                    #print(row)
                    return  # Skip writing if the row already exists
    
    # If we get here, it means the data_row doesn't exist in the file (or the file doesn't exist)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # If the file doesn't exist, write a header
            if len(data_row) == 3:
                writer.writerow(["Dicom", "Label","shape"])
            elif len(data_row) == 2:
                writer.writerow(["Dicom", "Label","shape"])
            else:
                writer.writerow(["Dicom"])
        writer.writerow(data_row)


##### end csv preprocessing ###
def correct_aspect(image_slice, pixel_spacing, slice_thickness, desired_aspect='COR'):
    """
    Corrects the aspect ratio for an image slice based on pixel spacing and slice thickness.
    
    :param image_slice: 2D numpy array of the image slice.
    :param pixel_spacing: Tuple of (pixel_spacing_x, pixel_spacing_y).
    :param slice_thickness: Thickness of the slice.
    :param desired_aspect: String, the desired aspect ratio (COR, SAG, or TRA).
    :return: Image slice with the corrected aspect ratio.
    """
    
    # Determine aspect ratios based on the desired view
    if desired_aspect== 'COR':
        # For COR view: adjust for the ratio of slice thickness to pixel spacing in x
        aspect_ratio = slice_thickness / pixel_spacing[0]
    elif desired_aspect== 'SAG':
        # For SAG view: adjust for the ratio of slice thickness to pixel spacing in y
        aspect_ratio =   pixel_spacing[1] / slice_thickness  
    elif desired_aspect == 'TRA':
        # For TRA view: adjust for the ratio of pixel spacing in y to pixel spacing in x
        aspect_ratio = pixel_spacing[1] / pixel_spacing[0]
    else:
        raise ValueError("Invalid desired_aspect. Choose from 'COR', 'SAG', or 'TRA'.")
    
    # Calculate the zoom factors
    zoom_factor = aspect_ratio if desired_aspect in ['COR', 'SAG'] else 1/aspect_ratio
    
    # # Apply the zoom to correct the aspect ratio
    # if desired_aspect in ['COR', 'SAG']:
    #     corrected_image = zoom(image_slice, (1, zoom_factor))
    # else:
    #     corrected_image = zoom(image_slice, (zoom_factor, 1))
    # Determine the number of dimensions of image_slice
    num_dims = image_slice.ndim

    # Apply the zoom to correct the aspect ratio
    # Adjust the zoom factor sequence based on the number of dimensions
    if desired_aspect in ['COR', 'SAG']:
        zoom_factors = (1,) * (num_dims - 1) + (zoom_factor,)
    else:
        zoom_factors = (zoom_factor,) + (1,) * (num_dims - 1)

    corrected_image = zoom(image_slice, zoom_factors, order=3)  # Using cubic interpolation
    
    return corrected_image


def get_orientation(affine):
    # Extract direction cosines
    R = affine[:3, :3]
    orientation = nib.aff2axcodes(R)
    return orientation

def get_pixel_spacing_and_slice_thickness_from_nifti(nifti_file):
    try:
        # Load the NIfTI file
        img = nib.load(nifti_file)
        header = img.header

        # Determine the valid affine matrix (sform or qform)
        affine = img.affine

        # Calculate the spacing for each dimension using the affine matrix
        spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

        # The slice thickness is typically the largest of the three, as MRI/CT scans often have a larger spacing between slices
        # Pixel spacing is the other two dimensions
        # This is a general assumption and might need adjustment for specific datasets
        slice_thickness = np.max(spacing)
        pixel_spacing_indices = np.argsort(spacing)[:2]  # Get indices of the two smaller dimensions
        pixel_spacing = spacing[pixel_spacing_indices]

        return pixel_spacing, slice_thickness

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    
def get_pixel_spacing_and_slice_thickness_from_nifti(nifti_file):
    try:
        # Load the NIfTI file
        img = nib.load(nifti_file)
        header = img.header

        # Extract voxel dimensions from pixdim
        # Skipping the first element as it's usually -1 or 1
        voxel_dims = header['pixdim'][1:4]

        # Pixel spacing is generally the first two values of pixdim
        pixel_spacing = voxel_dims[:2]

        # Slice thickness can be inferred from the third value of pixdim
        # But it's wise to verify this with the image's orientation and slice direction
        slice_thickness = voxel_dims[2]

        return pixel_spacing, slice_thickness

    except Exception as e:
        try:
            # Determine the valid affine matrix (sform or qform)
            affine = img.affine

            # Calculate the spacing for each dimension using the affine matrix
            spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

            # The slice thickness is typically the largest of the three, as MRI/CT scans often have a larger spacing between slices
            # Pixel spacing is the other two dimensions
            # This is a general assumption and might need adjustment for specific datasets
            slice_thickness = np.max(spacing)
            pixel_spacing_indices = np.argsort(spacing)[:2]  # Get indices of the two smaller sdimensions
            pixel_spacing = spacing[pixel_spacing_indices]
            return pixel_spacing, slice_thickness
        except:
            return None, None
            print("pixel spacing retrieval failed for"+nifti_file)

       
def get_pixel_spacing_and_slice_thickness(dicom_dataset):
    # Default values
    default_pixel_spacing = [1, 1]
    default_slice_thickness = 1

    # Attempt to retrieve Pixel Spacing and Slice Thickness
    # If not found or on error, default values are returned
    try:
        pixel_spacing = getattr(dicom_dataset, 'PixelSpacing', default_pixel_spacing)
        slice_thickness = getattr(dicom_dataset, 'SliceThickness', default_slice_thickness)

        # Sometimes Pixel Spacing and Slice Thickness might be stored under different tags
        # You can add conditions to check for other common tags as well
        if not pixel_spacing:
            pixel_spacing = dicom_dataset.get('00280030', default_pixel_spacing)  # Standard tag for Pixel Spacing
        if not slice_thickness:
            slice_thickness = dicom_dataset.get('00180050', default_slice_thickness)  # Standard tag for Slice Thickness

    except AttributeError:
        # In case the DICOM dataset is missing these attributes entirely
        pixel_spacing = default_pixel_spacing
        slice_thickness = default_slice_thickness

    return pixel_spacing, slice_thickness

from scipy.stats import mode



def get_background_value_from_3d_array(image_data, num_slices=3):
 
    # Take the first and last `num_slices` from the 3D array
    early_slices = image_data[:num_slices].flatten()
    late_slices = image_data[-num_slices:].flatten()

    # Combine them into a single array
    edge_slices = np.concatenate((early_slices, late_slices))

    # Compute the mode of the combined slices
    background_value = mode(edge_slices, keepdims=True).mode[0]
    return background_value

def resize_or_pad_image(selected_slice, target_size, background_value):
    
    current_height, current_width = selected_slice.shape
    desired_height, desired_width = target_size

    # Calculate resize factors for both dimensions
    resize_factor_y = desired_height / current_height
    resize_factor_x = desired_width / current_width

    # Determine which dimension is larger and resize
    if current_height > desired_height or current_width > desired_width:
        resize_factor = min(resize_factor_y, resize_factor_x)
        selected_slice = ndimage.zoom(selected_slice, (resize_factor, resize_factor), order=3)
    
    selected_slice[selected_slice<background_value] = background_value
    # After resizing, pad the smaller dimension if needed
    current_height, current_width = selected_slice.shape
    pad_height = (desired_height - current_height) if current_height < desired_height else 0
    pad_width = (desired_width - current_width) if current_width < desired_width else 0

    # Add padding symmetrically with the background pixel value
    selected_slice = np.pad(selected_slice, 
                            ((pad_height // 2, pad_height - pad_height // 2), 
                             (pad_width // 2, pad_width - pad_width // 2)),
                            'constant', constant_values=background_value)

    return selected_slice
import matplotlib.cm as cm

# def get_view_order(nifti_image_path):
#     # Load the NIfTI image
#     img = nib.load(nifti_image_path)
#     # Get the image dimensions (shape)
#     dimensions = img.header.get_data_shape()
    
#     # Assuming dimensions are in the order of (X, Y, Z)
#     # X - Sagittal, Y - Coronal, Z - Transverse
#     dim_order = sorted([(dim, label) for dim, label in zip(dimensions, ['sag', 'cor', 'tra'])], reverse=True)
    
#     # Sort based on dimensions and create a string representation of the order
#     view_order = '>'.join([label for _, label in dim_order])
    
#     return view_order

# def get_view_order(nifti_image_path):
#     # Load the NIfTI image
#     img = nib.load(nifti_image_path)
#     # Get the image dimensions (shape)
#     dimensions = img.header.get_data_shape()
#     # Get voxel sizes (physical dimensions of each voxel)
#     voxel_sizes = img.header.get_zooms()
    
#     # Combine the dimensions with their corresponding labels and voxel sizes
#     dim_info = [(dim, voxel_size, label) for dim, voxel_size, label in zip(dimensions, voxel_sizes, ['sag', 'cor', 'tra'])]
    
#     # Sort based on dimensions and voxel sizes, considering both size and physical dimension
#     # This considers the physical space the image occupies, which helps in distinguishing orientations better
#     dim_info_sorted = sorted(dim_info, key=lambda x: (x[0]*x[1], x[1]), reverse=True)
    
#     # Create a string representation of the order, including both anatomical label and physical dimension
#     view_order = '>'.join([f"{label} ({dim}x{voxel_size:.2f}mm)" for dim, voxel_size, label in dim_info_sorted])
    
#     return view_order
def get_view_order(nifti_image_path):
    
    # Load the NIfTI image
    img = nib.load(nifti_image_path)
    # Get the image dimensions (shape)
    dimensions = img.header.get_data_shape()
    # Get voxel sizes (physical dimensions of each voxel)
    voxel_sizes = img.header.get_zooms()
    
    # Combine the dimensions with their corresponding labels and voxel sizes
    dim_info = [(label, voxel_size) for _, voxel_size, label in zip(dimensions, voxel_sizes, ['sag', 'cor', 'tra'])]
    
    # No need to sort based on dimensions since we are focusing on orientation order and voxel sizes for logic
    return dim_info

def needs_swapping_cor_sag(dim_info, acquisition_matrix):
    # Extract voxel sizes for each orientation
    sag_voxel_size, cor_voxel_size, tra_voxel_size = (size for _, size in dim_info)

    # Calculate differences
    sag_cor_diff = abs(sag_voxel_size - cor_voxel_size)
    cor_tra_diff = abs(cor_voxel_size - tra_voxel_size)

    # Determine if coronal and transverse sizes are nearly identical
    cor_tra_near_identical = cor_tra_diff < 0.005
    sag_cor_large = sag_cor_diff > 0.38
    sag_cor_small = sag_cor_diff > 0.2  # Adjusted to check for small difference

    # Evaluate significant differences between sagittal and coronal/transverse sizes
    significant_difference = 0.015  # Threshold for considering significant size difference
    acquisition_matrix_non_zero = [x for x in acquisition_matrix if x != 0]
    identical_acquisition_matrix_values = (len(acquisition_matrix_non_zero) >= 2 and
                                           all(val == acquisition_matrix_non_zero[0] for val in acquisition_matrix_non_zero))

    # Conditions indicating a need for swapping
    condition1 = (sag_cor_diff >= significant_difference)  \
        and cor_tra_near_identical and not sag_cor_large and ((not identical_acquisition_matrix_values and sag_cor_small) or (identical_acquisition_matrix_values and not sag_cor_small))

    return condition1

def needs_swapping_cor_tra(dim_info,acquisition_matrix):
    # Extract voxel sizes for each orientation
    sag_voxel_size, cor_voxel_size, tra_voxel_size = (size for _, size in dim_info)
    

    # Calculate differences
    cor_tra_diff = abs(cor_voxel_size - tra_voxel_size)
    sag_tra_diff = abs(sag_voxel_size - tra_voxel_size)
    sag_cor_diff = abs(sag_voxel_size - cor_voxel_size)
    
    cor_tra_near_identical = sag_cor_diff < 0.005

    # Evaluate significant differences for determining swap need
    significant_diff_for_swap = 0.02  # A threshold to adjust based on observed discrepancies
    acquisition_matrix_non_zero = [x for x in acquisition_matrix if x != 0]
    identical_acquisition_matrix_values = (len(acquisition_matrix_non_zero) >= 2 and
                                            all(val == acquisition_matrix_non_zero[0] for val in acquisition_matrix_non_zero))
    # Conditions indicating a need for swapping
    # A significant difference between coronal and transverse or sagittal and transverse dimensions suggests a swap might be needed
    condition_for_swap = cor_tra_near_identical and sag_tra_diff > significant_diff_for_swap and identical_acquisition_matrix_values

    return condition_for_swap


def needs_swapping_3dsag_cor(dim_info,acquisition_matrix):
    # Extract voxel sizes for each orientation
    sag_voxel_size, cor_voxel_size, tra_voxel_size = (size for _, size in dim_info)

    # Calculate differences
    sag_cor_diff = abs(sag_voxel_size - cor_voxel_size)
    cor_tra_diff = abs(cor_voxel_size - tra_voxel_size)
    sag_tra_diff = abs(sag_voxel_size - tra_voxel_size)

    # Determine if coronal and transverse sizes are nearly identical
    sag_tra_near_identical = sag_tra_diff < 0.001
    sag_cor_near_identical = sag_cor_diff < 0.005
    sag_cor_near_large = sag_cor_diff > 0.2
    sag_cor_near_small = sag_cor_diff > 0.005 and  sag_cor_diff < 0.15
    # sag_cor_large = sag_cor_diff > 0.38
    # sag_cor_small = sag_cor_diff > 0.2  # Adjusted to check for small difference
    # nifty_or = get_nifti_orientation(a)
    # # Evaluate significant differences between sagittal and coronal/transverse sizes
    # significant_difference = 0.015  # Threshold for considering significant size difference
    # acquisition_matrix_non_zero = [x for x in acquisition_matrix if x != 0]
    condition2 = acquisition_matrix[1] == acquisition_matrix[2] and acquisition_matrix[0] != acquisition_matrix[1] and acquisition_matrix[2] != acquisition_matrix[3]
    # Conditions indicating a need for swapping
    # condition1 = (sag_cor_diff >= significant_difference)  \
    #     and cor_tra_near_identical and not sag_cor_large and ((not identical_acquisition_matrix_values and sag_cor_small) or (identical_acquisition_matrix_values and not sag_cor_small))

    return sag_tra_near_identical and not sag_cor_near_identical and (sag_cor_near_large or condition2)

import numpy as np

def check_orientation_axis_swap(iop):
    """
    Determines if axis swap is needed based on ImageOrientationPatient (IOP) DICOM attribute.
    
    Args:
        iop (list): The ImageOrientationPatient attribute, a list of six values.
    
    Returns:
        bool: True if axis swap is needed, False otherwise.
    """
    # Ensure iop is a numpy array with correct shape
    iop = np.array(iop)
    if iop.shape != (6,):
        raise ValueError("IOP must be a list or array of six elements.")
    
    # Extract row and column direction cosines from IOP
    row_cosine = iop[:3]
    col_cosine = iop[3:]
    
    # Calculate the cross product to get the slice normal vector
    normal_cosine = np.cross(row_cosine, col_cosine)
    
    # Define standard anatomical direction vectors
    directions = {
        'Sagittal': np.array([1, 0, 0]),
        'Coronal': np.array([0, 1, 0]),
        'Axial': np.array([0, 0, 1])
    }
    
    # Function to find the closest anatomical direction
    def closest_direction(vector):
        max_dot = -1
        closest_dir = ""
        for dir_name, dir_vector in directions.items():
            dot = np.dot(vector, dir_vector)
            if abs(dot) > max_dot:
                max_dot = abs(dot)
                closest_dir = dir_name
        return closest_dir, max_dot
    
    # Find the primary orientation based on the normal vector
    primary_orientation, _ = closest_direction(normal_cosine)
    
    # For secondary orientation, we check against the row vector
    secondary_orientation, _ = closest_direction(row_cosine)
    
    # Determine if an axis swap is needed based on primary and secondary orientations
    # This logic may need to be adjusted based on your specific criteria for swapping
    if primary_orientation == "Sagittal" and secondary_orientation == "Coronal" or primary_orientation == "Axial" and secondary_orientation == "Sagittal" :
        return True
    else:
        return False
    

def get_anatomical_orientation(iop):
    """
    Determine the anatomical orientation of the image based on the IOP vector.
    
    Args:
        iop (list): ImageOrientationPatient attribute, a list of six values.
    
    Returns:
        tuple: A tuple containing strings representing the primary, secondary, and slice normal orientations.
    """
    # Ensure IOP is a numpy array
    iop = np.array(iop).reshape(2, 3)  # Reshape to 2x3 for row and column vectors
    row_vector, col_vector = iop[0], iop[1]
    
    # Standard anatomical directions
    directions = {
        'Sagittal': np.array([1, 0, 0]),
        'Coronal': np.array([0, 1, 0]),
        'Axial': np.array([0, 0, 1])
    }
    
    # Calculate the cross product to find the normal vector
    normal_vector = np.cross(row_vector, col_vector)
    
    # Function to find the closest anatomical direction
    def find_closest_direction(vector):
        max_cosine = -np.inf
        closest_direction = None
        for name, dir_vector in directions.items():
            cosine_angle = np.dot(vector, dir_vector) / (np.linalg.norm(vector) * np.linalg.norm(dir_vector))
            if cosine_angle > max_cosine:
                max_cosine = cosine_angle
                closest_direction = name
        return closest_direction
    
    # Identify orientations
    row_orientation = find_closest_direction(row_vector)
    col_orientation = find_closest_direction(col_vector)
    normal_orientation = find_closest_direction(normal_vector)
    
    return (row_orientation, col_orientation, normal_orientation)


# def check_need_for_axis_swap(iop):
#     """
#     Check if there's a need to swap axes based on the anatomical orientations.
    
#     Args:
#         iop (list): ImageOrientationPatient attribute, a list of six values.
    
#     Returns:
#         bool: True if an axis swap is needed, False otherwise.
#     """
#     row_orientation, col_orientation, normal_orientation = get_anatomical_orientation(iop)
    
#     # Example logic for determining the need for an axis swap
#     # This is simplistic and should be adjusted based on specific requirements
#     if normal_orientation == 'Axial':
#         if :
#             return False  # No swap needed
#         else:
#             return True  # Swap might be needed
#     else:
#         # Add more conditions as needed based on the analysis or visualization requirements
#         return False





def anatomical_orientation(iop):
    """
    Determines the anatomical orientation and order based on Image Orientation Patient (IOP) DICOM attribute.
    
    Args:
        iop (list or np.ndarray): The Image Orientation Patient attribute, a list or array of six values.
        
    Returns:
        str: A string describing the anatomical orientation in Siemens-like format (e.g., "Sag>Cor>Tra").
    """
    # Ensure IOP is a NumPy array
    iop = np.array(iop).reshape(2, 3)
    row_cosine, col_cosine = iop
    
    # Define standard anatomical direction vectors
    standard_planes = {
        'Sagittal': np.array([1, 0, 0]),
        'Coronal': np.array([0, 1, 0]),
        'Axial': np.array([0, 0, 1])
    }
    
    # Calculate the normal vector to the image plane
    normal_cosine = np.cross(row_cosine, col_cosine)
    
    def angle_between(v1, v2):
        """Calculate the angle in degrees between two vectors."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        return np.degrees(angle)
    
    # Calculate angles between the image plane normal and standard anatomical planes
    angles = {plane: angle_between(normal_cosine, vector) for plane, vector in standard_planes.items()}
    
    # Determine primary, secondary, and tertiary orientations based on the angles
    sorted_orientations = sorted(angles.items(), key=lambda x: x[1])
    
    if sorted_orientations[0][0] == 'Axial' and sorted_orientations[1][0] == 'Coronal' and sorted_orientations[2][0] == 'Sagittal':
        return True
    # # Construct a Siemens-like orientation string
    # orientation_str = '>'.join([f"{plane}({angle:.1f})" for plane, angle in sorted_orientations])
    
    # return orientation_str



def switch_cor_sag(nifti_path, output_path):
    # Load the NIfTI image
    nifti_image = nib.load(nifti_path)

    # Get the image data array
    image_data = nifti_image.get_fdata()

    # Check if image is 3D
    if len(image_data.shape) != 3:
        raise ValueError("The NIfTI image is not 3D and cannot be processed with this script.")

    # Switch the coronal (Y) and sagittal (X) axes
    # In NumPy, the axes are typically (sagittal, coronal, axial) or (X, Y, Z)
    switched_data = np.swapaxes(image_data, 0, 1)

    # Create a new NIfTI image with the switched data
    # It's important to also apply the same permutation to the affine matrix to keep the spatial information consistent
    new_affine = nifti_image.affine[:, [1, 0, 2, 3]]
    switched_image = nib.Nifti1Image(switched_data, affine=new_affine)

    # Save the new NIfTI image
    nib.save(switched_image, output_path)


def dicom_to_png_by_orientation(root_dir_or_dicomfolders, output_dir, desired_orientation="COR",target_size = (224, 224), plot_rgb=False):
    """
    Converts DICOM images to PNG by orientation, adjusting the aspect ratio based on pixel spacing and slice thickness.

    :param root_dir: The root directory where DICOM files are located.
    :param output_dir: The directory where PNG files will be saved.
    :param desired_orientation: The desired orientation ('COR', 'TRA', 'SAG') for the output image.
    :param pixel_spacing: The pixel spacing (x, y) from the DICOM metadata.
    :param slice_thickness: The slice thickness from the DICOM metadata.
    """
    not_4d = []
    failed = []
    is_4d = []
    if not isinstance(root_dir_or_dicomfolders, list):
    
        ignore_list = ["CR","US","XA","CR","PHS","PX","DX",
                       "RTDOSE", "RTSTRUCT", "RTIMAGE", 
                       "RTRECORD", "RTPLAN", "PR", "SR", "SC"]
        dicom_folders = [x for x in glob.glob(root_dir_or_dicomfolders + '/*/*/*/*/*/*') if os.path.isdir(x) and x.split('/')[-2] not in ignore_list]
    else:
        dicom_folders = root_dir_or_dicomfolders
    for dicom_folder in dicom_folders:
       
        modality = dicom_folder.split('/')[-2]
                    
        ###initilase###
        rgb = False
        ## define the output folder
        path_parts = dicom_folder.split('/')
        output_folder = output_dir + '/' + '/'.join(path_parts[-4:-1])
        output_filename = '_'.join(path_parts[-6:])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        orientation = dicom_folder.split("/")[-4]
        if os.path.isfile(output_folder + '/' + output_filename +'.png'):
            continue
        # Adjust for additional orientations like '3DCOR', '3DSAG', etc.
        if '3D' in orientation or 'NA' in orientation:
            orientation = "TRA"
        elif 'SAG' in orientation:
            orientation = 'SAG'
        elif 'COR' in orientation:
            orientation = 'COR'
        elif 'TRA' in orientation:
            orientation = 'TRA'           
        ##  load
        nii_d = glob.glob(dicom_folder + '.nii.gz')
        if not nii_d:
            convert_image(dicom_folder)
            nii_d = glob.glob(dicom_folder + '.nii.gz')
            if not nii_d:
                print("no nifti for dicom_folder"+dicom_folder)  
                continue
        try:
            img = nib.load(nii_d[0])
            image_data_init = img.get_fdata()
        except:
            try:
                if is_rgb(nii_d[0])[0]:
                    pil_img = get_rgb_image(nii_d[0], orientation, desired_orientation="COR",plot=False)
                    pil_img.save(output_folder + '/' + output_filename + '.png')
                    continue
            except:
                print("error"+dicom_folder)
        if not nii_d:
            convert_image(dicom_folder)
            if not nii_d:
                continue
        pixel_spacing,slice_thickness =  get_pixel_spacing_and_slice_thickness_from_nifti(nii_d[0])
        if pixel_spacing is None or len(pixel_spacing) == 0 or slice_thickness is None:
            ds = pydicom.read_file(glob.glob(dicom_folder+'/*.dcm')[0])
            pixel_spacing,slice_thickness =  get_pixel_spacing_and_slice_thickness(ds)
          

        selected_slice = None
        try:
            nifty_or = get_nifti_orientation(nii_d)
        except:
            nifty_or = "NS"
        
        if "4D" in modality:
            if len(image_data_init.shape) >3:
                channels = image_data_init.shape[-1]
            else:
                if len(image_data_init.shape) >3:
                    channels = image_data_init.shape[-1]
                    is_4d.append(dicom_folder)
                else:
                    channels = 1
        else:
            channels = 1
        for c in range(channels):
            if channels ==  1:
                image_data = image_data_init
            else:
                image_data = image_data_init[:,:,:,c]
            if nifty_or == "neurological":
            
                 # Selecting the appropriate slice based on the desired orientation
                 if desired_orientation == "COR":
                     if orientation == "SAG":
                         selected_slice = image_data[image_data.shape[0]//2,:,:]
                     elif orientation == "TRA":
                         selected_slice = image_data[:,image_data.shape[1]//2,:]
                     else:
                         selected_slice = image_data[:,:,image_data.shape[2]//2]
                 elif desired_orientation == "TRA":
                     if orientation == "SAG":
                         selected_slice = image_data[:,image_data.shape[1]//2,:]
                     elif orientation == "TRA":
                         selected_slice = image_data[:,:,image_data.shape[2]//2]
                     else:
                         selected_slice = image_data[:,image_data.shape[1]//2,:]
                 elif desired_orientation == "SAG":
                     if orientation == "SAG":
                         selected_slice = image_data[:,:,image_data.shape[2]//2]
                     elif orientation == "TRA" :
                         selected_slice = image_data[image_data.shape[0]//2,:,:]
                     else:
                         selected_slice = image_data[image_data.shape[0]//2,:,:]
            else:
                 if desired_orientation == "COR":
                     if orientation == "SAG":
                         selected_slice = image_data[:,image_data.shape[1]//2,:]
                     elif orientation == "TRA":
                         selected_slice = image_data[:,:,image_data.shape[2]//2]
                     else:
                         selected_slice = image_data[:,image_data.shape[1]//2,:]
                         
                 elif desired_orientation == "TRA":
                     if orientation == "SAG":
                         selected_slice = image_data[:,:,image_data.shape[2]//2]         
                     elif orientation == "TRA":
                         selected_slice = image_data[:,:,image_data.shape[2]//2]
                     else:
                         selected_slice = image_data[:,:,image_data.shape[2]//2]           
                 elif desired_orientation == "SAG":
                     if orientation == "SAG":
                         selected_slice = image_data[image_data.shape[0]//2,:,:]         
                     elif orientation == "TRA":
                         selected_slice = image_data[:,image_data.shape[1]//2,]
                     else:
                         selected_slice = image_data[image_data.shape[0]//2,:,:]
                 
            if "CT" not in modality and "PT" not in modality:
                selected_slice[selected_slice<0] = 0
                 
     
            # # Selecting the appropriate slice based on the desired orientation
            # if desired_orientation == "COR":
            #     if orientation == "SAG":
            #         selected_slice = image_data[image_data.shape[0]//2,:]
            #     elif orientation == "TRA":
            #         selected_slice = image_data[:,image_data.shape[1]//2,:]
            #     else:
            #         selected_slice = image_data[:,:,image_data.shape[2]//2]
            # elif desired_orientation == "TRA":
            #     if orientation == "SAG":
            #         selected_slice = image_data[:,image_data.shape[1]//2,:]
            #     elif orientation == "TRA":
            #         selected_slice = image_data[:,:,image_data.shape[2]//2]
            #     else:
            #         selected_slice = image_data[:,image_data.shape[1]//2,:]
            # elif desired_orientation == "SAG":
            #     if orientation == "SAG":
            #         selected_slice = image_data[:,:,image_data.shape[2]//2]
            #     elif orientation == "TRA" :
            #         selected_slice = image_data[image_data.shape[0]//2,:,:]
            #     else:
            #         selected_slice = image_data[image_data.shape[0]//2,:,:]
    
            # Check if selected_slice is None or not 2D
            if selected_slice is None or len(selected_slice.shape) != 2:
                print(f"No valid slice selected for folder: {dicom_folder}")
                failed.append(dicom_folder)
                continue
        
            # Correcting the aspect ratio
            if orientation != desired_orientation:
                selected_slice = correct_aspect(selected_slice, pixel_spacing, slice_thickness)
        
            # Resizing the selected slice to a fixed size (128x128 pixels)
            try:
                resized_slice = resize_or_pad_image(selected_slice, target_size,
                                                    get_background_value_from_3d_array(image_data, num_slices=3))
            except:
                print("ERROR in resizing"+dicom_folder)
        
            # Normalizing and saving the image
            resized_slice = np.interp(resized_slice, (resized_slice.min(), resized_slice.max()), (0, 255))
            pil_img = Image.fromarray(resized_slice.astype(np.uint8), mode='L')
            if orientation == "SAG" and desired_orientation == "COR" or orientation == "COR" and desired_orientation == "SAG" :
                pil_img = Image.fromarray(resized_slice.astype(np.uint8), mode='L').rotate(180)
            if orientation ==  desired_orientation or orientation == "TRA" and desired_orientation == "COR" or orientation == "TRA" and desired_orientation == "SAG" :
                pil_img = Image.fromarray(resized_slice.astype(np.uint8), mode='L').rotate(90)
            if orientation == "COR" and desired_orientation == "TRA" :
               pil_img = Image.fromarray(resized_slice.astype(np.uint8), mode='L').rotate(-90)
            
            if modality == "PT" and plot_rgb:
    
            # Apply the jet colormap
                colormap_image = cm.jet(resized_slice / 255)  # Normalize the image to range [0, 1]
                colormap_image = np.uint8(colormap_image * 255)  # Convert to 8-bit format
                pil_img = Image.fromarray(colormap_image, mode='RGBA')
            
            if channels !=1:
                pil_img.save(output_folder + '/' + output_filename +'_'+str(c)+ '.png')
            else:
                pil_img.save(output_folder + '/' + output_filename +'.png')
            
    return is_4d, not_4d, failed
    
# def extract_slice(image_data, orientation, desired_orientation, index=0):
#     """ Extracts a specific slice from the image data. """
#     if desired_orientation == "COR":
#         if orientation == "SAG":
#             selected_slice = image_data[image_data.shape[0]//2 + index,:]
#         elif orientation == "TRA":
#             selected_slice = image_data[:,image_data.shape[1]//2 + index,:]
#         else:
#             selected_slice = image_data[:,:,image_data.shape[2]//2 + index]
#     elif desired_orientation == "TRA":
#         if orientation == "SAG":
#             selected_slice = image_data[:,image_data.shape[1]//2 + index,:]
#         elif orientation == "TRA":
#             selected_slice = image_data[:,:,image_data.shape[2]//2 + index]
#         else:
#             selected_slice = image_data[:,image_data.shape[1]//2 + index,:]
#     elif desired_orientation == "SAG":
#         if orientation == "SAG":
#             selected_slice = image_data[:,:,image_data.shape[2]//2 + index]
#         elif orientation == "TRA":
#             selected_slice = image_data[image_data.shape[0]//2 + index,:,:]
#         else:
#             selected_slice = image_data[image_data.shape[0]//2 + index,:,:]
#     return selected_slice

# def dicom_to_png_by_orientation2(root_dir_or_dicomfolders, output_dir, desired_orientation="COR", 
#                                  target_size=(224, 224), plot_rgb=False, plot_quantiles=False, 
#                                  num_slices_to_plot=10, plot_in_single_figure=True):
#     """
#     Converts DICOM images to PNG by orientation, adjusting the aspect ratio based on pixel spacing and slice thickness,
#     and optionally plots quantiles or all slices.

#     :param plot_all_slices: Boolean indicating whether to plot all slices in a single file.
#     """

#     if not isinstance(root_dir_or_dicomfolders, list):
#         ignore_list = ["RTDOSE", "RTSTRUCT", "RTIMAGE", "RTRECORD", "RTPLAN", "PR", "SR"]
#         dicom_folders = [x for x in glob.glob(root_dir_or_dicomfolders + '/*/*/*/*/*/*') if os.path.isdir(x) and x.split('/')[-2] not in ignore_list]
#     else:
#         dicom_folders = root_dir_or_dicomfolders

#     for dicom_folder in dicom_folders:
#         modality = dicom_folder.split('/')[-2]
#         path_parts = dicom_folder.split('/')
#         output_folder = output_dir + '/' + '/'.join(path_parts[-4:-1])
#         output_filename = '_'.join(path_parts[-6:])
        
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder, exist_ok=True)

#         orientation = path_parts[-4].split("_")[0]  # Assuming format like '3DSAG' or 'COR'

#         # Load NIfTI data
#         nii_d = glob.glob(dicom_folder + '.nii.gz')
#         if not nii_d:
#             continue  # Skip if no NIfTI file found
#         img = nib.load(nii_d[0])
#         image_data_init = img.get_fdata()
        
#         pixel_spacing, slice_thickness = get_pixel_spacing_and_slice_thickness_from_nifti(nii_d[0])

#         if "4D" in modality:
#             channels = image_data_init.shape[-1]
#         else:
#             channels = 1
            
#         middle_index = image_data_init.shape[2] // 2
#         slices_to_plot = num_slices_to_plot // 2
#         start_slice = max(middle_index - slices_to_plot, 0)
#         end_slice = min(middle_index + slices_to_plot, image_data_init.shape[2])

#         if plot_in_single_figure:
#             # Plotting in a single figure for each slice (if 4D, create subplots for each channel)
#             for idx in range(start_slice, end_slice):
#                 fig, axes = plt.subplots(1, channels, figsize=(channels * 8, 8))
#                 if channels == 1:
#                     axes = [axes]  # Ensure axes is always a list

#                 for c in range(channels):
#                     slice_img = extract_slice(image_data_init[:,:,:,c] if channels > 1 else image_data_init, orientation, desired_orientation, index=idx - middle_index)
#                     ax = axes[c]
#                     ax.imshow(slice_img, cmap='gray' if modality not in ['PT', 'ANGIO', '4DPER'] else 'jet')
#                     ax.axis('off')
#                     ax.set_title(f'Channel {c+1}')

#                 plt.tight_layout()
#                 slice_filename = f"{output_filename}_slice_{idx}.png"
#                 plt.savefig(os.path.join(output_folder, slice_filename), bbox_inches='tight', pad_inches=0)
#                 plt.close(fig)

#         else:
#             for c in range(channels):
#                 selected_slice = extract_slice(image_data_init if channels == 1 else image_data_init[:,:,:,c], orientation, desired_orientation)
#                 if selected_slice is None or len(selected_slice.shape) != 2:
#                     print(f"No valid slice selected for folder: {dicom_folder}")
#                     #continue

#                 # Aspect ratio correction, resizing, and normalization
#                 selected_slice = correct_aspect(selected_slice, pixel_spacing, slice_thickness)
#                 resized_slice = resize_or_pad_image(selected_slice, target_size, get_background_value_from_3d_array(image_data_init, num_slices=3))
#                 resized_slice = np.interp(resized_slice, (resized_slice.min(), resized_slice.max()), (0, 255))

#                 # Image rotation and color mapping
#                 pil_img = Image.fromarray(resized_slice.astype(np.uint8), mode='L')
#                 if modality == "PT" and plot_rgb:
#                     colormap_image = cm.jet(resized_slice / 255)
#                     colormap_image = np.uint8(colormap_image * 255)
#                     pil_img = Image.fromarray(colormap_image, mode='RGBA')

#                 if plot_quantiles:
#                     # Plotting quantiles
#                     quantile_indices = [int(image_data_init.shape[2] * q) - (1 if q == 1 else 0) for q in [0, 0.1, 0.25, 0.5, 0.75, 1]]
#                     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#                     axes_flat = axes.flatten()
#                     for i, idx in enumerate(quantile_indices):
#                         slice_img = extract_slice(image_data_init if channels == 1 else image_data_init[:,:,:,c], orientation, desired_orientation, index=idx)
#                         axes_flat[i].imshow(slice_img, cmap='gray' if modality not in ['PT', 'ANGIO', '4DPER'] else 'jet')
#                         axes_flat[i].axis('off')
#                     plt.tight_layout()
#                     plt.savefig(output_folder + '/' + output_filename + '_quantiles.png')
#                     plt.close(fig)
#                 else:
#                     pil_img.save(output_folder + '/' + output_filename + '.png')


import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

def get_middle_slice(image_data, orientation, desired_orientation):
    """
    Extract the middle slice of an image based on the orientation.

    :param image_data: The 3D or 4D numpy array of the image data.
    :param orientation: The orientation of the image ('COR', 'TRA', 'SAG').
    :param desired_orientation: The desired orientation for the output image.
    :return: The middle slice or slices as a numpy array.
    """
    if len(image_data.shape) == 3:  # 3D image
        image_data = np.expand_dims(image_data, axis=-1)
        channel_indices = 0

    num_channels = image_data.shape[-1]
    if num_channels > 5:
        selected_percentiles = [1, 50, 99]
        channel_indices = [int(num_channels * p / 100) for p in selected_percentiles]
    else:
        channel_indices = range(num_channels)

    slices = []
    for c in channel_indices:
        # Selecting the appropriate slice based on the desired orientation
        # Selecting the appropriate slice based on the desired orientation
        if desired_orientation == "COR":
            if orientation == "SAG":
                selected_slice = image_data[image_data.shape[0]//2,:,:,c]
            elif orientation == "TRA":
                selected_slice = image_data[:,image_data.shape[1]//2,:,c]
            else:
                selected_slice = image_data[:,:,image_data.shape[2]//2,c]
        elif desired_orientation == "TRA":
            if orientation == "SAG":
                selected_slice = image_data[:,image_data.shape[1]//2,:,c]
            elif orientation == "TRA":
                selected_slice = image_data[:,:,image_data.shape[2]//2,c]
            else:
                selected_slice = image_data[:,image_data.shape[1]//2,:,c]
        elif desired_orientation == "SAG":
            if orientation == "SAG":
                selected_slice = image_data[:,:,image_data.shape[2]//2,c]
            elif orientation == "TRA" :
                selected_slice = image_data[image_data.shape[0]//2,:,:,c]
            else:
                selected_slice = image_data[image_data.shape[0]//2,:,:,c]

        slices.append(selected_slice)

    return np.stack(slices, axis=-1)

def create_subplot(slices, target_size, output_path):
    """
    Create and save a subplot from the given slices.

    :param slices: The numpy array of slices.
    :param target_size: The target size for each subplot image.
    :param output_path: The path to save the subplot image.
    """
    num_channels = slices.shape[-1]
    fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 3, 3))
    if num_channels == 1:
        axes = [axes]
    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(slices[:,:,i], cmap='gray', interpolation='nearest')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# Your existing function with modifications
def dicom_to_png_by_orientation3(root_dir_or_dicomfolders, output_dir, 
                                 desired_orientation="COR", target_size=(224, 224), plot_rgb=True):
    
    if not isinstance(root_dir_or_dicomfolders, list):
        ignore_list = ["RTDOSE", "RTSTRUCT", "RTIMAGE", "RTRECORD", "RTPLAN", "PR", "SR"]
        dicom_folders = [x for x in glob.glob(root_dir_or_dicomfolders + '/*/*/*/*/*/*') if os.path.isdir(x) and x.split('/')[-2] not in ignore_list]
    else:
        dicom_folders = root_dir_or_dicomfolders
    failed_list = []
    not_4d = []
    for dicom_folder in dicom_folders:
        print(dicom_folder)
        if dicom_folder == "/e210/HD8/GLIOMA/all_pids/1530131/20170614/TRA/HNC/4DPER/GREEPIPERFTRA5MM2S_GREEPIPERFTRA5MM2S_114888":
            continue
        modality = dicom_folder.split('/')[-2]
        path_parts = dicom_folder.split('/')
        output_folder = output_dir + '/' + '/'.join(path_parts[-4:-1])
        output_filename = '_'.join(path_parts[-6:])
        if os.path.isfile(output_folder + '/' + output_filename + '.png'):
            continue
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        orientation = path_parts[-4].split("_")[0]  # Assuming format like '3DSAG' or 'COR'
        # Adjust for additional orientations like '3DCOR', '3DSAG', etc.
        if '3D' in orientation:
            orientation = "TRA"
        elif 'SAG' in orientation:
            orientation = 'SAG'
        elif 'COR' in orientation:
            orientation = 'COR'
        elif 'TRA' in orientation:
            orientation = 'TRA'           
            
        nii_d = glob.glob(dicom_folder + '.nii.gz')
        if not nii_d:
            convert_image(dicom_folder)
            nii_d = glob.glob(dicom_folder + '.nii.gz')
            if not nii_d:
                print("no nifti for dicom_folder"+dicom_folder)  
                continue
        try:
            img = nib.load(nii_d[0])
            image_data = img.get_fdata()
        except:
            try:
                if is_rgb(nii_d[0])[0]:
                    pil_img = get_rgb_image(nii_d[0], orientation, desired_orientation="COR",plot=False)
                    pil_img.save(output_folder + '/' + output_filename + '.png')
                    continue
            except:
                print("error"+dicom_folder)
        try:
            pixel_spacing,slice_thickness =  get_pixel_spacing_and_slice_thickness_from_nifti(nii_d[0])
            if pixel_spacing is None or len(pixel_spacing) == 0 or slice_thickness is None:
                pixel_spacing,slice_thickness =  get_pixel_spacing_and_slice_thickness(ds)
        except:
            failed_list.append(dicom_folder)
            continue
        if "4D" in modality:
            if not len(image_data.shape) >3:
                not_4d.append(dicom_folder)
  
        # Extract the middle slice or slices
        try:
            selected_slices = get_middle_slice(image_data, orientation, desired_orientation)
        except:
            print(" I failed before"+dicom_folder)
            failed_list.append(dicom_folder)
            continue
        # Check if selected_slices is None or not valid
        if selected_slices is None or len(selected_slices.shape) < 3:
            print(f"No valid slice selected for folder: {dicom_folder}")
            failed_list.append(dicom_folder)
            continue

         # Handle single slice or multiple slices (subplot)
    
        if selected_slices.shape[-1] == 1:
            
            selected_slice = selected_slices[:,:,0]
            pil_img = get_slice(image_data,modality,selected_slice,orientation,desired_orientation,
                                pixel_spacing,slice_thickness,target_size,plot_rgb)

            pil_img.save(output_folder + '/' + output_filename +'.png')
        else:
            # Resize, normalize, and save as subplot
            resized_slices = []
            for c in range(selected_slices.shape[-1]):
                try:
                    pil_img = get_slice(image_data,modality,selected_slices[:,:,c],orientation,desired_orientation,
                                        pixel_spacing,slice_thickness,target_size, plot_rgb)
                    resized_slices.append(pil_img)
                    failed = False
                except:
                    print("ERROR in resizing"+dicom_folder)
                    
                    failed = True

            if failed:
                failed_list.append(dicom_folder)
                continue
            resized_slices = np.stack(resized_slices, axis=-1)
            output_path = output_folder + '/' + output_filename + '.png'
            create_subplot(resized_slices, target_size, output_path)

    return not_4d, failed_list



def get_slice(image_data_init,modality,selected_slice,orientation,desired_orientation,
              pixel_spacing,slice_thickness,target_size, plot_rgb):
    # Correcting the aspect ratio
    if orientation != desired_orientation:
        selected_slice = correct_aspect(selected_slice, pixel_spacing, slice_thickness)


    # Resizing the selected slice to the target size
    #try:
    resized_slice = resize_or_pad_image(selected_slice, target_size,
                                        get_background_value_from_3d_array(image_data_init, num_slices=3))
    # except:
    #     print("ERROR in resizing"+dicom_folder)
     

    # Normalizing and saving the image
    resized_slice = np.interp(resized_slice, (resized_slice.min(), resized_slice.max()), (0, 255))
    pil_img = Image.fromarray(resized_slice.astype(np.uint8), mode='L')
    if orientation == "SAG" and desired_orientation == "COR" or orientation == "COR" and desired_orientation == "SAG" :
        pil_img = Image.fromarray(resized_slice.astype(np.uint8), mode='L').rotate(180)
    if orientation ==  desired_orientation or orientation == "TRA" and desired_orientation == "COR" or orientation == "TRA" and desired_orientation == "SAG" :
        pil_img = Image.fromarray(resized_slice.astype(np.uint8), mode='L').rotate(90)
    if orientation == "COR" and desired_orientation == "TRA" :
       pil_img = Image.fromarray(resized_slice.astype(np.uint8), mode='L').rotate(-90)
    
    if modality == "PT" and plot_rgb:

    # Apply the jet colormap
        colormap_image = cm.jet(resized_slice / 255)  # Normalize the image to range [0, 1]
        colormap_image = np.uint8(colormap_image * 255)  # Convert to 8-bit format
        pil_img = Image.fromarray(colormap_image, mode='RGBA')
    return pil_img



def delete_from_png(png_dir,dataset_dir):

    dicom_folders = []
    for f in glob.glob(png_dir + '/*'):
        try:
            dicom = get_dicom_from_png(f,dataset_dir)[0]
            dicom_folders.append(dicom)
        except:
            print(f)
    for f in dicom_folders:
        converted_files = glob.glob(f +'*.ni*')
        for c in converted_files:
            os.remove(c)
        shutil.rmtree(f)

def replot_corrupt(png_dir,dataset_dir):

    dicom_folders = []
    for f in glob.glob(png_dir + '/*'):
        dicom = get_dicom_from_png(f,dataset_dir)[0]
        convert_image(dicom)
        dicom_folders.append(dicom)
    dicom_to_png_by_orientation(dicom_folders,dataset_dir+'_'+os.path.basename(png_dir),
                                desired_orientation="COR",plot_rgb=True)


def reorient(img):
    cmd = ("fslreorient2std {0} {1}".format(img, img))
    sp.check_output(cmd, shell=True) 

def get_rgb_image(nii_d, orientation,desired_orientation="COR",plot=False):
    # Read the image using SimpleITK
    img = sitk.ReadImage(nii_d)
    
    # Convert the image to a numpy array
    image_data = sitk.GetArrayFromImage(img)
    
    spacing = img.GetSpacing()  # Returns a tuple (x_spacing, y_spacing, z_spacing)
    pixel_spacing = (spacing[0], spacing[1])  # Pixel spacing in x and y
    slice_thickness = spacing[2]  # Slice thickness (z spacing)
    
    # Select the middle slice for 'COR' or 'SAG' orientations
    # Assuming the image_data has shape [slices, height, width, channels]
    if desired_orientation == "COR":
        if orientation == "SAG":
            selected_slice = image_data[image_data.shape[0]//2,:,:]
        elif orientation == "TRA":
            selected_slice = image_data[:,image_data.shape[1]//2,:,:]
        else:
            selected_slice = image_data[image_data.shape[0]//2,:,:]
    elif desired_orientation == "TRA":
        if orientation == "SAG":
            selected_slice = image_data[:,image_data.shape[1]//2,:,:]
        elif orientation == "TRA":
            selected_slice = image_data[:,:,image_data.shape[2]//2,:]
        else:
            selected_slice = image_data[:,image_data.shape[1]//2,:,:]
    elif desired_orientation == "SAG":
        if orientation == "SAG":
            selected_slice = image_data[:,:,image_data.shape[2]//2,:]
        elif orientation == "TRA" :
            selected_slice = image_data[image_data.shape[0]//2,:,:,:]
        else:
            selected_slice = image_data[image_data.shape[0]//2,:,:,:]
    if orientation != desired_orientation:
        selected_slice = correct_aspect(selected_slice, pixel_spacing, slice_thickness, desired_aspect='COR')
    # # Plot the slice - assuming the image is in RGB
    # # Transpose the array to have the channels as the last dimension for plotting
    # slice_to_plot = slice_to_plot.transpose(1, 2, 0) if desired_orientation != "TRA" else slice_to_plot

    # Show the image
    if plot:
        plt.imshow(slice_to_plot)
        plt.axis('off')  # Hide the axes
        plt.show()
    pil_img = Image.fromarray(selected_slice.astype(np.uint8), mode='RGB')
    if orientation == "COR":
        pil_img = pil_img.rotate(180)
    return pil_img


def dicom_to_png_by_orientation_single(dicom_folder,desired_orientation="COR",target_size = (224, 224)):
    """
    Converts DICOM images to PNG by orientation, adjusting the aspect ratio based on pixel spacing and slice thickness.

    :param root_dir: The root directory where DICOM files are located.
    :param output_dir: The directory where PNG files will be saved.
    :param desired_orientation: The desired orientation ('COR', 'TRA', 'SAG') for the output image.
    :param pixel_spacing: The pixel spacing (x, y) from the DICOM metadata.
    :param slice_thickness: The slice thickness from the DICOM metadata.
    """

   
    nii_d = glob.glob(dicom_folder + '.nii.gz')
    dcm_file = glob.glob(dicom_folder + '/*.dcm')[0]
    ds = pydicom.read_file(dcm_file)
    rgb = False
    orientation = dicom_folder.split("/")[-4]
    # Adjust for additional orientations like '3DCOR', '3DSAG', etc.
    if '3D' in orientation or 'NA' in orientation:
        orientation = "TRA"
    elif 'SAG' in orientation:
        orientation = 'SAG'
    elif 'COR' in orientation:
        orientation = 'COR'
    elif 'TRA' in orientation:
        orientation = 'TRA'        
    try:
        img = nib.load(nii_d[0])
        image_data = img.get_fdata()
    except:
        if is_rgb(nii_d[0])[0]:
            pil_img = get_rgb_image(nii_d,orientaion, desired_orientation="COR",plot=True)
            #return

    
    pixel_spacing,slice_thickness =  get_pixel_spacing_and_slice_thickness_from_nifti(nii_d[0])
    if pixel_spacing is None or len(pixel_spacing) == 0 or slice_thickness is None:
        pixel_spacing,slice_thickness =  get_pixel_spacing_and_slice_thickness(ds)
 
    selected_slice = None
    
    nifty_or = get_nifti_orientation(nii_d[0])
    
    if nifty_or != "neurological":
        # Selecting the appropriate slice based on the desired orientation
        if desired_orientation == "COR":
            if orientation == "SAG":
                selected_slice = image_data[image_data.shape[0]//2,:,:]
            elif orientation == "TRA":
                selected_slice = image_data[:,image_data.shape[1]//2,:]
            else:
                selected_slice = image_data[:,:,image_data.shape[2]//2]
        elif desired_orientation == "TRA":
            if orientation == "SAG":
                selected_slice = image_data[:,image_data.shape[1]//2,:]
            elif orientation == "TRA":
                selected_slice = image_data[:,:,image_data.shape[2]//2]
            else:
                selected_slice = image_data[:,image_data.shape[1]//2,:]
        elif desired_orientation == "SAG":
            if orientation == "SAG":
                selected_slice = image_data[:,:,image_data.shape[2]//2]
            elif orientation == "TRA" :
                selected_slice = image_data[image_data.shape[0]//2,:,:]
            else:
                selected_slice = image_data[image_data.shape[0]//2,:,:]
    else:
        if desired_orientation == "COR":
            if orientation == "SAG":
                selected_slice = image_data[:,image_data.shape[1]//2,:]
            elif orientation == "TRA":
                selected_slice = image_data[image_data.shape[0]//2,:,]
            else:
                selected_slice = image_data[:,image_data.shape[1]//2,:]
                
        elif desired_orientation == "TRA":
            if orientation == "SAG":
                selected_slice = image_data[:,:,image_data.shape[2]//2]         
            elif orientation == "TRA":
                selected_slice = image_data[:,:,image_data.shape[2]//2]
            else:
                selected_slice = image_data[:,:,image_data.shape[2]//2]           
        elif desired_orientation == "SAG":
            if orientation == "SAG":
                selected_slice = image_data[:,:,image_data.shape[2]//2]         
            elif orientation == "TRA":
                selected_slice = image_data[:,image_data.shape[1]//2,]
            else:
                selected_slice = image_data[image_data.shape[0]//2,:,:]
        
        selected_slice[selected_slice<0] = 0
    # Check if selected_slice is None or not 2D
    if selected_slice is None or len(selected_slice.shape) != 2:
        print(f"No valid slice selected for folder: {dicom_folder}")
        return

    # Correcting the aspect ratio
    if orientation != desired_orientation:
        selected_slice = correct_aspect(selected_slice, pixel_spacing, slice_thickness)

    # Resizing the selected slice to a fixed size (128x128 pixels)
    try:
        resized_slice = resize_or_pad_image(selected_slice, target_size,
                                            get_background_value_from_3d_array(image_data, num_slices=3))
    except:
        print("ERROR in resizing"+dicom_folder)

    # Normalizing and saving the image
    resized_slice = np.interp(resized_slice, (resized_slice.min(), resized_slice.max()), (0, 255))
    if rgb:
        mode_method = "RGB"
    else:
        mode_method = "L"
    pil_img = Image.fromarray(resized_slice.astype(np.uint8), mode=mode_method)
    
    if orientation == "SAG" and desired_orientation == "COR" or orientation == "COR" and desired_orientation == "SAG" :
        pil_img = pil_img.rotate(180)
    if orientation ==  desired_orientation or orientation == "TRA" and desired_orientation == "COR" or orientation == "TRA" and desired_orientation == "SAG" :
        pil_img = pil_img.rotate(90)
    if orientation == "COR" and desired_orientation == "TRA" :
       pil_img = pil_img.rotate(-90)
       
    plt.imshow(pil_img, cmap='gray')  # 'gray' colormap for grayscale
    plt.axis('off')  # Hide the axes
    plt.show()

def get_nifti_orientation(nifti_path):
    # Load the NIFTI file
    image = nib.load(nifti_path)
    affine = image.affine

    # Determine the direction of the x-axis in the voxel space
    # A positive x direction in the second column of the affine matrix indicates a 'neurological' orientation
    # A negative x direction indicates a 'radiological' orientation
    orientation = 'neurological' if affine[0, 0] > 0 else 'radiological'

    return orientation
    
def numeric_sort_key(filepath):
    base_name = os.path.basename(filepath)
    parts = base_name.split('.dcm')
    numeric_part = parts[0] + '.' + parts[1] if len(parts) > 1 else parts[0]
    return float(numeric_part)



# def order_dicom_files(dicom_folder):
#     dicom_files = glob.glob(dicom_folder + '/*.dcm')
#     att = "SOPInstanceUID"
#     ds = pydicom.dcmread(dicom_files[0],force=True)
#     sl = getattr(ds, "SliceLocation", None)
#     if not sl:
#         sl1 = getattr(ds, "ImagePositionPatient", None)
#         if sl1:
#             ds2 = pydicom.dcmread(dicom_files[1],force=True)
#             sl2 = getattr(ds2, "ImagePositionPatient", None)
#             differing_dim =  [i for i, (v1, v2) in enumerate(zip(sl1, sl2)) if v1 != v2]
#             if len (differing_dim)==1:
#                 att = "ImagePositionPatient"
#                 dim = differing_dim[0]    
#     else:
#         att = "SliceLocation"
#     index = 1
#     for dicom_file in dicom_files:
#         ds = pydicom.dcmread(dicom_file,force=True)
#         sl = getattr(ds, att, "NS")
#         if att == "ImagePositionPatient":
#             sl = sl[dim]
#         new_file_name = str(sl) + ".dcm"
#         new_file_path = os.path.join(dicom_folder, new_file_name)
#         # Check if file exists, and if so, append _index to avoid overwriting
#         while os.path.exists(new_file_path):
#             new_file_name = str(sl) + "_" + str(index) + ".dcm"
#             new_file_path = os.path.join(dicom_folder, new_file_name)
#             index += 1
#         os.rename(dicom_file, new_file_path)
#     rename_files_numerically(dicom_folder)
#     dicom_files = glob.glob(dicom_folder + '/*.dcm')
#     dicom_files.sort(key=numeric_sort_key)
#     return dicom_files

    

# def order_dicom_files(dicom_folder):
#     dicom_files = glob.glob(dicom_folder + '/*.dcm')
#     # Read all the ImagePositionPatient attributes
#     image_positions = [pydicom.dcmread(dicom_file, force=True).ImagePositionPatient for dicom_file in dicom_files]
    
#     # Convert the attributes to numpy array for easier manipulation
#     positions = np.array([list(map(float, pos)) for pos in image_positions])

#     # Compute the difference between first two slices to get the slicing direction
#     direction = np.cross(positions[1] - positions[0], positions[2] - positions[0])

#     # Compute the dot product with the direction to project the positions along that direction
#     ordering = np.dot(positions, direction)

#     # Sort the files by the projected positions
#     sorted_files = [dicom_files[i] for i in np.argsort(ordering)]
    
#     # Rename the files to make their names reflect their order
#     for i, file_path in enumerate(sorted_files):
#         new_file_name = str(i) + ".dcm"
#         new_file_path = os.path.join(dicom_folder, new_file_name)
#         os.rename(file_path, new_file_path)

#     return sorted_files

def order_dicom_files(dicom_folder_or_files):
    
    if isinstance(dicom_folder_or_files, str):  # It's a folder path
        dicom_files = glob.glob(dicom_folder_or_files + '/*.dcm')
        dicom_folder = dicom_folder_or_files
    elif isinstance(dicom_folder_or_files, list):  # It's a list of DICOM files
        dicom_files = dicom_folder_or_files
        dicom_folder = os.path.dirname(dicom_files[0])
    else:
        raise ValueError("Expected a folder path or a list of DICOM files.")

    slice_positions = []

    for file in dicom_files:
        ds = pydicom.dcmread(file, force=True)
        try:
            image_orientation,att = get_image_orientation(ds)
        except:
       
            image_orientation,att = None, None
        # if image_orientation is None:
        #     return dicom_files
        image_position = np.array(ds.ImagePositionPatient, dtype=float)
        image_orientation = np.array(image_orientation, dtype=float)
        #patient_position = getattr(ds, 'PatientPosition', None)
        patient_position = ds.PatientPosition
        
        row_orientation = image_orientation[:3]
        col_orientation = image_orientation[3:]
        slice_normal = np.cross(row_orientation, col_orientation)

        if patient_position in ['HFP', 'FFP', 'FFS', 'HFS']:
            slice_normal = -slice_normal

        slice_position = np.dot(slice_normal, image_position)  # Same as before
        slice_positions.append(slice_position)

    # Sort the files by the computed slice positions
    sorted_files = [dicom_files[i] for i in np.argsort(slice_positions)]
    

    # Rename the files based on the computed ordering
    for i, file_path in enumerate(sorted_files):
        new_index = i + 1  # Start numbering from 1
        new_file_name = str(new_index).zfill(4) + ".dcm"  # Using zfill to keep consistent numbering
        new_file_path = os.path.join(dicom_folder, new_file_name)
        os.rename(file_path, new_file_path)
        
    dicom_files = glob.glob(dicom_folder + '/*.dcm')
    dicom_files.sort(key=numeric_sort_key)
    
    if len(dicom_files) != len(sorted_files):
        raise ValueError("After ordering slices deleted")
    # After sorting, renaming the files from 1 to number of dcm files
    for i, file_path in enumerate(dicom_files):
        new_file_name = str(i + 1) + ".dcm"  # renaming files starting from 1.dcm
        new_file_path = os.path.join(dicom_folder, new_file_name)
        os.rename(file_path, new_file_path)
    
    # Now get the sorted and renamed dicom_files
    dicom_files = glob.glob(dicom_folder + '/*.dcm')
    dicom_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))  # now sorting by the new numeric names
        
    return dicom_files

def label_img_diddict(target_directory, modality_dict):
    dicom_files = glob.glob(os.path.join(target_directory, '**', '*.dcm'), recursive=True)
    cmr_to_non_c_list = []
    mr_to_c_list = []

    for dicom_file in dicom_files:
        dcm = pydicom.dcmread(dicom_file)
        sop_instance_uid = dcm.SOPInstanceUID
        old_name = os.path.basename(os.path.dirname(dicom_file))

        modality = None
        for key, uids in modality_dict.items():
            if sop_instance_uid in uids:
                modality = key
                break

        if modality:
            # Create the destination directory path
            relative_path = os.path.relpath(dicom_file, target_directory)
            parts = relative_path.split(os.sep)

            # Update the old modality to the new one
            parts[-3] = modality  # Now changing the parent's folder name

            destination_path = os.path.join(target_directory, os.sep.join(parts))

            destination_dir = os.path.dirname(destination_path)

            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

            shutil.move(dicom_file, destination_path)

            # Check for the conditions to add to lists
            if old_name == "CMR" and not modality.startswith("C") and modality != "DCE":
                cmr_to_non_c_list.append(dicom_file)
            elif old_name == "MR" and (modality.startswith("C") or modality == "DCE") and modality != "CT":
                mr_to_c_list.append(dicom_file)

        else:
            print(f"No modality found for SOPInstanceUID: {sop_instance_uid}")


    return cmr_to_non_c_list, mr_to_c_list





# def order_dicom_files(dicom_folder):
#     dicom_files = glob.glob(dicom_folder + '/*.dcm')
#     slice_positions = []

#     for file in dicom_files:
#         ds = pydicom.dcmread(file, force=True)
#         image_position = np.array(ds.ImagePositionPatient, dtype=float)
#         image_orientation = np.array(ds.ImageOrientationPatient, dtype=float)
#         patient_position = ds.PatientPosition

#         row_orientation = image_orientation[:3]
#         col_orientation = image_orientation[3:]
#         slice_normal = np.cross(row_orientation, col_orientation)

#         if patient_position in ['HFP', 'FFP', 'FFS', 'HFS']:
#             slice_normal = -slice_normal

#         slice_position = np.dot(slice_normal, image_position)  # Same as before
#         slice_positions.append(slice_position)

#     # Sort the files by the computed slice positions
#     sorted_files = [dicom_files[i] for i in np.argsort(slice_positions)]

#     # Rename the files based on the computed ordering
#     for i, file_path in enumerate(sorted_files):
#         new_file_name = str(i).zfill(4) + ".dcm" # Using zfill to keep consistent numbering
#         new_file_path = os.path.join(dicom_folder, new_file_name)
#         os.rename(file_path, new_file_path)
#     dicom_files = glob.glob(dicom_folder + '/*.dcm')
#     dicom_files.sort(key=numeric_sort_key)
#     return dicom_files
# import cv2
# def datatype_to_depth_flag(dtype):
#     if dtype == np.uint8:
#         return cv2.CV_8U
#     elif dtype == np.uint16:
#         return cv2.CV_16U
#     elif dtype == np.int16:
#         return cv2.CV_16S
#     # ... add other datatypes as needed
#     else:
#         raise ValueError(f"Unsupported datatype: {dtype}")
        
# def remove_background_markings(file_dcm, save_path=None, overwrite=False):
#     # 1. Load the DICOM file
#     dcm_file = pydicom.dcmread(file_dcm)

#     # 2. Convert DICOM data to an image
#     img = dcm_file.pixel_array
#     # Normalize to 0-255 for standard image operations
#     img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#     # 3. Image Processing
#     # Apply a binary threshold to separate markings from the main image
#     _, thresholded = cv2.threshold(img_normalized, 127, 255, cv2.THRESH_BINARY_INV)

#     # Use morphological operations to further clean the image or remove small markings
#     kernel = np.ones((5,5), np.uint8)
#     cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    
#     # Check if there are any markings
#     if np.any(cleaned):
#         print("Markings detected in the DICOM image.")
#         # Merge the cleaned background with the original image
#         final_image = cv2.bitwise_and(img_normalized, img_normalized, mask=cleaned)
#     else:
#         print("No markings detected in the DICOM image.")
#         final_image = img_normalized

#     # Optionally, save the result
#     if save_path:
#         cv2.imwrite(save_path, final_image)
    
#     # If overwrite is True, update the DICOM pixel data and save the DICOM file
#     if overwrite:
#         # Convert the processed image back to its original range and datatype
#         depth_flag = datatype_to_depth_flag(img.dtype)
#         final_image = cv2.normalize(final_image, None, img.min(), img.max(), cv2.NORM_MINMAX, dtype=depth_flag)
#         dcm_file.PixelData = final_image.tobytes()
#         dcm_file.save_as(file_dcm)

#     return final_image, img_normalized
def check_split_multiple_echo_times(dicom_folder):
    dicom_dict = defaultdict(list)
    tr = None
    sequence_type = None
    name = dicom_folder.split('/')[-1]

    for file in os.listdir(dicom_folder):
        if file.endswith('.dcm'):
            filepath = os.path.join(dicom_folder, file)
            ds = pydicom.dcmread(filepath)
            
            te = getattr(ds, "EchoTime", None)
            if te is not None:
                dicom_dict[str(te)].append(filepath)
            
            if tr is None:
                tr = getattr(ds, "RepetitionTime", None)
    sequence_type = determine_sequence_type(filepath)
    if len(dicom_dict) > 2:
        return "ME"
    if len(dicom_dict) == 2:
        if "VAI" in ds.SeriesDescription.upper():
            return "VAI"
        te1, te2 = sorted(dicom_dict.keys())
        files1, files2 = dicom_dict[te1], dicom_dict[te2]

        if sequence_type == "GE":
            if tr and tr < 600:
                short_te_folder, long_te_folder = 'T1IP', 'T1OP'
                ll = "T1IPOP"
            else:
                short_te_folder, long_te_folder = 'PD', 'T2S'
                ll = "PDT2S"
        else:
            short_te_folder, long_te_folder = 'PD', 'T2'
            ll = "PDT2"

        short_te_folder_path = os.path.join(os.path.dirname(os.path.dirname(dicom_folder)),short_te_folder+'-'+sequence_type,name)
        long_te_folder_path = os.path.join(os.path.dirname(os.path.dirname(dicom_folder)), long_te_folder+'-'+sequence_type,name)

        os.makedirs(short_te_folder_path, exist_ok=True)
        os.makedirs(long_te_folder_path, exist_ok=True)

        for file in files1:
            shutil.move(file, short_te_folder_path)
        for file in files2:
            shutil.move(file, long_te_folder_path)
        conv_image, cont = convert_image(short_te_folder_path)
        conv_image, cont = convert_image(long_te_folder_path)
        os.rmdir(dicom_folder)
        os.remove(dicom_folder+'.nii.gz')
        return ll

    
# def check_split_multiple_echo_times(dicom_folder,output_dir):
#     # Create a dictionary to store the DICOM files by echo time
#     dicom_dict = defaultdict(list)
#     name = dicom_folder.split('/')[-1]
    
#     # Initialize the repetition time variable
#     tr = None

#     # Iterate over the DICOM files in the directory
#     for file in os.listdir(dicom_folder):
#         if file.endswith('.dcm'):
#             filepath = os.path.join(dicom_folder, file)
#             ds = pydicom.dcmread(filepath)

#             # Get the Echo Time (TE) from the DICOM header
#             te = str(getattr(ds, "EchoTime",""))
#             if te !="":
#                 dicom_dict[te].append(filepath)
#             if tr is None:
#                 tr = getattr(ds, "RepetitionTime", None)

#     # Check the number of different echo times
#     if len(dicom_dict) > 2:
#         return "ME"
#     elif len(dicom_dict) == 2:
#         if "VAI" in ds.SeriesDescription.upper():
#             return "VAI"
#         if "T1" in ds.SeriesDescription.upper() or "T2" in ds.SeriesDescription.upper() or "GRME" in ds.SeriesDescription.upper() or "PD" in ds.SeriesDescription.upper():
#             # Find the two echo times and corresponding file lists
#             te1, te2 = dicom_dict.keys()
#             files1, files2 = dicom_dict.values()
    
#             # Determine which echo time is shorter
#             if te1 < te2:
#                 short_te_files, long_te_files = files1, files2
#             else:
#                 short_te_files, long_te_files = files2, files1
#             if tr is not None and tr < 1000:  # This threshold may need adjusting based on your specific case
#                 short_te_folder, long_te_folder = 'T1IP', 'T1OP'
#                 ll = "T1IPOP"
#             else:
#                 short_te_folder, long_te_folder = 'PD', 'T2'
#                 ll = "PDT2"
#             # Create the folders for the short and long TE files
#             short_te_folder_path = os.path.join(os.path.dirname(os.path.dirname(dicom_folder)),short_te_folder,name)
#             long_te_folder_path = os.path.join(os.path.dirname(os.path.dirname(dicom_folder)), long_te_folder,name)
    
#             os.makedirs(short_te_folder_path, exist_ok=True)
#             os.makedirs(long_te_folder_path, exist_ok=True)
    
#             # Move the files to their respective folders
#             for file in short_te_files:
#                 shutil.move(file, short_te_folder_path)
#             for file in long_te_files:
#                 shutil.move(file, long_te_folder_path)
#             conv_image, cont = convert_image(short_te_folder_path)
#             conv_image, cont = convert_image(long_te_folder_path)
#             os.rmdir(dicom_folder)
#             os.remove(dicom_folder+'.nii.gz')
#             return ll
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
def dicom_to_png(root_dir, output_dir, modality_considered=modality_considered, reorder_folders=False, plot_quantiles=True):
    ignore_list = ["RTDOSE", "RTSTRUCT", "RTIMAGE", "RTRECORD", "RTPLAN", "PR", "SR"]
    colorjet_modalities = ["PT","ANGIO","4DPER"]
    dicom_folders = [x for x in glob.glob(root_dir + '/*/*/*/*/*/*') if os.path.isdir(x) and x.split('/')[-2] not in ignore_list]

    for dicom_folder in dicom_folders:
        modality = dicom_folder.split('/')[-2]
        path_parts = dicom_folder.split('/')
        output_folder = output_dir + '/' + '/'.join(path_parts[-4:-1])
        output_filename = '_'.join(path_parts[-6:])

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        if reorder_folders:
            try:
                dicom_files = order_dicom_files(dicom_folder)
            except:
                dicom_files = glob.glob(dicom_folder + '/*.dcm') 
        else:
            dicom_files = glob.glob(dicom_folder + '/*.dcm') 
            #dicom_files.sort(key=numeric_sort_key)
         
        try:
            if plot_quantiles:
                quantile_indices = [int(len(dicom_files) * q) - (1 if q == 1 else 0) for q in [0, 0.1, 0.25, 0.5, 0.75,1]]
                fig, axes = plt.subplots(2, 3, figsize=(15, 15))
                axes_flat = axes.flatten()  # Flatten the axes array
                for i, idx in enumerate(quantile_indices):
                    ds = pydicom.dcmread(dicom_files[idx])
                    img = ds.pixel_array
                    if img.ndim == 3:
                        img = img[:, :, 1]
                    img = normalize_zscore(img)
                    if modality not in colorjet_modalities:
                        axes_flat[i].imshow(img, cmap='gray')  # Use the flattened axes array
                    else:
                        axes_flat[i].imshow(img, cmap='jet')  # Use the flattened axes array
                    axes_flat[i].axis('off')
                plt.tight_layout()
                plt.savefig(output_folder + '/' + output_filename + '.png')
                plt.close()  # Explicitly close the figure to free up resources
            else:
                # Plotting only the middle slice
                middle_file = dicom_files[len(dicom_files) // 2]
                ds = pydicom.dcmread(middle_file)
                img = ds.pixel_array
                if img.ndim == 3:
                    img = img[:, :, 0]
                img = normalize_zscore(img)
                # Now, instead of directly saving the image through PIL, we'll use matplotlib to apply the colormap.
                plt.figure(figsize=(8,8))
                if modality not in colorjet_modalities:
                    plt.imshow(img, cmap='gray')
                else:
                    plt.imshow(img, cmap='jet')
                plt.axis('off')
                plt.savefig(output_folder + '/' + output_filename + '.png', bbox_inches='tight', pad_inches=0)
                plt.close()  
                # Explicitly close the figure to free up resources
                    # pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
                    # pil_img.save(output_folder + '/' + output_filename + '.png')

        except Exception as e:
            plt.close()
            print(f"Failed to save PNG for {dicom_folder}. Error: {e}")
            
def normalize_zscore(image):
    # Calculate mean and standard deviation
    mean = np.mean(image)
    std = np.std(image)
    # Normalize the image using z-score normalization
    normalized = (image - mean) / std
    # Rescale values to the range [0, 255]
    return np.interp(normalized, (normalized.min(), normalized.max()), (0, 255))


def adjust_label_from_png(root_dir,dataset_dir,index):
    classes = glob.glob(root_dir+'/*')
    for c in classes:
        files = glob.glob(c+'/*')
        for f in files:
            name = f.split('/')[-1]
            
            parts = name.split('_')
            parts = parts[:5]+['_'.join(parts[5:])]
        
            dicom_dir = os.path.join(dataset_dir, *parts).replace(".png","")
            
            modify_and_move(dicom_dir, index, c.split('/')[-1], replace=True)

def relabel_bodypart(dataset_dir):
    dirs = [ x for x in glob.glob(dataset_dir+'/*/*/*/*/*/*') if os.path.isdir(x)]
    for img_dir in dirs:
        file_dcm = glob.glob(img_dir+'/*.dcm')[0]
        bp = get_bodypart(ds)
        ds = pydicom.read_file(file_dcm)
        if bp != img_dir.split('/')[-3]:
            print("relabel"+img_dir)
            modify_and_move(img_dir, -3, bp , replace=True)

def modify_and_move(dicom_dir, index, name, replace=True):
    # Split the path into directories
    dirs = dicom_dir.split(os.sep)

    # Check if index is an integer or a string
    if isinstance(index, int):
        # If it's an integer, treat it as an index
        if replace:
            dirs[index] = name
        else:
            dirs.insert(index, name)
    else:
        # If it's a string, treat it as a directory name
        try:
            dir_index = dirs.index(index)
            dirs[dir_index] = name
        except ValueError:
            print(f"The directory name {index} does not exist in the path.")

    # Join the directories back into a path
    new_path = os.path.dirname(os.sep.join(dirs))

    # Move the folder to the new path
    move_folders_to_new_dir([dicom_dir,dicom_dir+'.nii.gz'], new_path)
    return os.path.join(new_path,dicom_dir.split('/')[-1])


# subset_dicom_to_png(root_dir, '/media/pgsalome/HD7/pancreas_retro_cohort_PsCurated_png_cor', "COR", -4, modality_considered=modality_considered)
# subset_dicom_to_png(root_dir, '/media/pgsalome/HD7/pancreas_retro_cohort_PsCurated_png_sag', "SAG", -4, modality_considered=modality_considered)
# subset_dicom_to_png(root_dir, '/media/pgsalome/HD7/pancreas_retro_cohort_PsCurated_png_ns', "NS", -4, modality_considered=modality_considered)
# subset_dicom_to_png(r, r+'_tra', "NS", -4, modality_considered=modality_considered)

def subset_dicom_to_png(root_dir, output_dir, name, index, modality_considered=modality_considered):
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Construct the path pattern
    path_parts = ['*'] * 6
    path_parts[index] = name

    # Join path_parts to create the glob pattern
    path_pattern = root_dir + '/' + '/'.join(path_parts)

    # Find matching folders
    dicom_folders = [x for x in glob.glob(path_pattern) if os.path.isdir(x) and any(word in x.split('/')[-2] for word in modality_considered)]

    for dicom_folder in dicom_folders:
        # Define the output filename
        path_parts = dicom_folder.split('/')
        output_filename = '_'.join(path_parts[-6:])

        dicom_files = glob.glob(dicom_folder + '/*.dcm')
        dicom_files.sort()  # make sure they're in order
        try:
            # Read the middle file
            middle_file = dicom_files[len(dicom_files) // 2]
            ds = pydicom.dcmread(middle_file)

            # Convert to numpy array and normalize to 0-255
            img = ds.pixel_array
            if img.ndim == 3:
                img = img[:, :, 1]
            img = np.interp(img, (img.min(), img.max()), (0, 255))

            # Convert to PIL image and save as PNG
            pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
            pil_img.save(output_dir + '/' + output_filename + '.png')
        except Exception as e:
            print(f"Failed to save PNG for {dicom_folder}. Error: {e}")



def get_idict_rd(root_dir):
    imae_dict = {}
    

    # use glob to get all subdirectories
    all_subdirs =[x for x in glob.glob(root_dir + '/*/*/*/*/*/*') if os.path.isdir(x)]
    for subdir in all_subdirs:
        parts = subdir.split(os.sep)
        group, subgroup, subsubgroup = parts[-4], parts[-3], parts[-2]
        imae_dict.setdefault(group, {}).setdefault(subgroup, {}).setdefault(subsubgroup, []).append(subdir)
        
    return imae_dict


def get_lists_idict(dictionary, key):
    image_list = []

    # If key is found at the first level
    if key in dictionary:
        sub_dict = dictionary.get(key, {})
        # If the sub_dict's values are lists, then flatten and return them.
        if all(isinstance(val, list) for val in sub_dict.values()):
            lists_of_paths = list(sub_dict.values())
            return list(itertools.chain(*lists_of_paths))
        # If the sub_dict's values are dictionaries, then go deeper to fetch the paths.
        else:
            for _, inner_dict in sub_dict.items():
                if isinstance(inner_dict, dict):
                    lists_of_paths = list(inner_dict.values())
                    image_list.extend(list(itertools.chain(*lists_of_paths)))

    # If key is not found at the first level, search the second level
    else:
        for _, dict_key in dictionary.items():
            for sub_key, dict_subkey in dict_key.items():
                if key == sub_key:
                    lists_of_paths = list(dict_subkey.values())
                    image_list.extend(list(itertools.chain(*lists_of_paths)))
                # If the key is not found at the second level, search the third level
                else:
                    for subsub_key, dict_subsubkey in dict_subkey.items():
                        if key == subsub_key:
                            if isinstance(dict_subsubkey, list):
                                image_list.extend(dict_subsubkey)
                            elif isinstance(dict_subsubkey, dict):
                                lists_of_paths = list(dict_subsubkey.values())
                                image_list.extend(list(itertools.chain(*lists_of_paths)))

    return image_list  # Return an empty list if the key is not found





def label_directories(root_dir,keyword,class_label,csv_file):
    matches = []
    
    for root, dirnames, filenames in os.walk(root_dir):
        for dirname in fnmatch.filter(dirnames, '*'+keyword+'*'):
            matches.append(os.path.join(root, dirname))
    for match in matches:
        new_dir = root_dir + '/'+ match.split('/')[5] + '/' + match.split('/')[6] + '/' + class_label
        os.makedirs(new_dir, exist_ok=True)
        shutil.move(match,new_dir)
        update_csv(csv_file,new_dir+'/'+match.split('/')[-1], class_label)


def move_folders(folders, subdirectory_name="HNC"):
    
    # Move each folder in the list to the subdirectory
    for folder in folders:
        # Get the parent directory of the first folder in the list
        parent_dir = os.path.dirname(os.path.dirname(folder))
        subdirectory_path = os.path.join(parent_dir, subdirectory_name)
        if not os.path.exists(subdirectory_path):
            os.makedirs(subdirectory_path,exist_ok=True)
        # Ensure that the folder exists before trying to move it
        if os.path.exists(folder):
            shutil.move(folder, os.path.join(subdirectory_path, os.path.basename(folder)))
        else:
            print(f"Folder {folder} does not exist, so it cannot be moved.")

def rename_dirs(root_dir, old_name, new_name):
    existing_dirs = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == old_name:
                old_dir_path = os.path.join(dirpath, dirname)
                new_dir_path = os.path.join(dirpath, new_name)

                if os.path.exists(new_dir_path):
                    existing_dirs.append(new_dir_path)
                    #raise Exception(f"Directory {new_dir_path} already exists!")
                else:
                    os.rename(old_dir_path, new_dir_path)
                    print(f"Renamed directory: {old_dir_path} to {new_dir_path}")

    return existing_dirs


def flatten_dict_of_lists(dict_of_lists):
    flattened_list = [item for sublist in dict_of_lists.values() for item in sublist]
    return flattened_list

def print_non_dcm_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith('.dcm'):
                print(os.path.join(root, file))


def remove_files_with_suffix(path, suffixes=[".json",".nii.gz",".bval",".nii",".bvec",".nii",".nrrd"]):
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.endswith(suffix) for suffix in suffixes):
                file_to_remove = os.path.join(root, file)
                print(f"Deleting: {file_to_remove}")
                os.remove(file_to_remove)


def get_folder_names(folder_list, index=2):
    # Initialize list for folder names
    folder_names = []
    for path in folder_list:
        folder_names.append(path.split('/')[-index])
    return folder_names

def extract_volume_from_4d_nifti(nifti_path, index, output_path):
    # Load the NIfTI file
    nifti_image = nib.load(nifti_path)

    # Check if the image is 4D
    if len(nifti_image.shape) != 4:
        print("The provided image is not 4D")
        return

    # Check if the index is within the range
    if index < 0 or index >= nifti_image.shape[3]:
        print("Index is out of range")
        return

    # Extract the 3D volume specified by the index
    extracted_volume_data = nifti_image.dataobj[..., index]

    # Create a new NIfTI image object with the extracted volume and the original header
    extracted_volume_image = nib.Nifti1Image(extracted_volume_data, nifti_image.affine, nifti_image.header)

    # Save the new NIfTI file
    nib.save(extracted_volume_image, output_path)
    print(f"Saved the extracted volume to {output_path}")


# def get_varying_dicom_attributes(dicom_folder):
    
#     ignore = ["WindowWidth", "WindowCenter", "SOPInstanceUID",
#               "PixelData", "LargestImagePixelValue", "InStackPositionNumber",
#               "InstanceNumber", "ContentTime", "ImagePositionPatient", "InstanceCreationTime", "SliceLocation"]

#     # Get all DICOM files in the directory
#     dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith(".dcm")]

#     # Create a dictionary to store the unique values of each DICOM attribute
#     attribute_values = defaultdict(set)

#     # For each DICOM file, get the attribute values and add them to the dictionary
#     for dicom_file in dicom_files:
#         # Read the DICOM file
#         ds = pydicom.dcmread(os.path.join(dicom_folder, dicom_file))

#         # For each attribute in the DICOM dataset, add its value to the dictionary
#         for element in ds.iterall():
#             tag = str(element.tag)
#             name = element.name   # Using tag for private attributes
#             full_name = f"{name} ({tag})"  # Combining name and tag
            
#             value = str(element.value) if element.value else None
            
#             # Check if the attribute is to be ignored
#             if full_name not in ignore and value is not None:
#                 attribute_values[full_name].add(value)

#     # Create a list to store the names and tags of the attributes with varying values
#     varying_attributes = [attribute for attribute, values in attribute_values.items() if len(values) > 1]

#     return varying_attributes
def get_lists_ndict(dictionary, key):
    image_list = []
    # If key is found at the first level
    if key in dictionary:
        sub_dict = dictionary.get(key, {})
        lists_of_paths = list(sub_dict.values())
        return list(itertools.chain(*lists_of_paths))

    # If key is not found at the first level, search the second level
    for _, dict_key in dictionary.items():
        for sub_key, dict_subkey in dict_key.items():
            if key == sub_key:
                for subsub_key, dict_subsubkey in dict_subkey.items():
                    image_list.extend(dict_subsubkey)


    return image_list  # Return an empty list if the key is not found

def get_diddict_rd(root_directory):
    """
    Create a dictionary with modalities as keys and lists of SOPInstanceUIDs as values.
    
    Args:
    - root_directory (str): The root directory to start the search for DICOM files.
    
    Returns:
    - dict: A dictionary where the keys are modalities and the values are lists of SOPInstanceUIDs.
    """
    modality_dict = {}
    
    # Search for all DICOM files in the root directory
    dicom_files = glob.glob(os.path.join(root_directory, '**', '*.dcm'), recursive=True)

    for dicom_file in dicom_files:
        # Read the DICOM file
        dcm = pydicom.dcmread(dicom_file)

        # Get the SOPInstanceUID
        sop_instance_uid = dcm.SOPInstanceUID
        
        # Determine the modality from the grandparent directory's name
        grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(dicom_file)))

        # Update the dictionary
        if grandparent_dir not in modality_dict:
            modality_dict[grandparent_dir] = []
        modality_dict[grandparent_dir].append(sop_instance_uid)
    
    return modality_dict

def check_4d_il(dicom_folders):
    """
    Processes the given DICOM folders through the conversion_function and checks the matrix shape in nibabel.
    
    Parameters:
    - dicom_folders: list of DICOM folder paths.
    - conversion_function: function to convert DICOM folder.
    
    Returns:
    - List of DICOM folders that don't result in a matrix shape of 4 when loaded with nibabel.
    """
    
    # List to hold DICOM folders that don't meet the matrix shape condition
    non_compliant_folders = []
    failed = []
    
    for folder in dicom_folders:
        print(folder)
        converted_file = folder+'.nii.gz'
        if not os.path.isfile(converted_file):
            # Convert DICOM folder using the provided conversion function
            try:
                conv_image, cont = convert_image(folder)
            except:
                failed.append(folder)
                continue                
            if not cont:
                failed.appemd(folder)
                continue
                
        # Load the converted output in nibabel
        img = nib.load(converted_file)
        data = img.get_fdata()
        
        # Check the shape of the data
        if len(data.shape) != 4:
            non_compliant_folders.append(folder)
            
    return non_compliant_folders

import random
def get_varying_dicom_attributes(dicom_folder):
    
    ignore_tags = ['(0043, 102a)', '(0051, 100a)','(0029, 1010)','(0008, 0018)', '(0008, 0033)', '(0019, 10a2)', '(0008, 0013)', '(0008, 0032)', '(0008, 1150)', 
                   '(0008, 2112)', '(0008, 0001)', '(0020, 0013)', '(0020, 1041)', '(0008, 0100)', '(0008, 0102)',
                   '(0008, 1155)', '(0028, 1101)', '(0028, 1102)', '(0028, 0030)', '(0040, 9216)', '(0040, 9211)', 
                   '(0040, 9096)', '(0028, 2112)', '(0018, 1020)', '(0020, 0032)', '(0020, 0000)', '(0020, 9057)', 
                   '(0027, 1041)', '(0028, 0107)', '(0008, 0104)', '(0028, 0106)', '(0020, 0037)', '(0028, 1103)',
                   '(0028, 1201)', '(0020, 0052)', '(0013, 1013)', '(0013, 1010)', '(0018, 1250)', '(0018, 1050)',
                   '(0008, 0031)', '(0008, 1010)', '(0018, 0020)', '(0018, 0050)', '(0018, 0083)', '(0018, 0088)',
                   '(0018, 1310)', '(0020, 0012)', '(0027, 1043)', '(0027, 1049)', '(0027, 104c)', '(0019, 10bb)', 
                   '(0019, 10bc)', '(0019, 10bd)', '(0028, 1050)', '(0028, 1051)', '(7fe0, 0010)', '(0027, 1040)', 
                   '(0008, 2111)', '(0028, 1203)', '(0028, 1202)', '(0028, 0000)', '(0040, 0000)', '(0018, 0000)', 
                   '(0029, 1131)', '(0027, 1042)', '(0027, 1044)', '(0027, 1048)', '(0027, 104a)', '(0027, 104b)', 
                   '(0027, 104d)', '(0027, 1035)', '(0028, 0101)', '(0028, 0102)', '(0020, 0050)', '(0028, 0031)', 
                   '(0028, 0032)', '(0008, 0000)','(0028, 1055)','(0020, 0030)','(0020, 0035)','(0040, 0275)',
                   '(2001, 100a)','(0018,9089)','(2001, 100b)','(0019, 100b)','(0029, 1031)','(0018, 1014)']

   

    # Get all DICOM files in the directory
    dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith(".dcm") ]
    if len(dicom_files)>200:
        dicom_files = random.sample(dicom_files, 100)

    # Create a dictionary to store the unique values of each DICOM attribute
    attribute_values = defaultdict(set)
    # Variable to keep track of ImagePosition values
    image_position_values = set()
    # Initialize varying_attributes outside the loop to avoid the UnboundLocalError
    varying_attributes = []    

    # For each DICOM file, get the attribute values and add them to the dictionary
    for dicom_file in dicom_files:
        # Read the DICOM file
        ds = pydicom.dcmread(os.path.join(dicom_folder, dicom_file))

        # Check for duplicate ImagePosition values
        ImagePositionPatient = getattr(ds, 'ImagePositionPatient', None)
        if ImagePositionPatient:
            ImagePositionPatient = tuple(ImagePositionPatient)
            if ImagePositionPatient in image_position_values:

                attribute_values["Image Position (Patient) ((0020, 0032))"].add(ImagePositionPatient)
            image_position_values.add(ImagePositionPatient)
            
        # For each attribute in the DICOM dataset, add its value to the dictionary
        for element in ds.iterall():
            tag = str(element.tag)
            
            # Check if the tag is to be ignored
            if tag in ignore_tags:
                continue
            
            name = element.name  # Using tag for private attributes
            full_name = f"{name} ({tag})"  # Combining name and tag
            
            value = str(element.value) if element.value else None
            
            if value is not None:
                attribute_values[full_name].add(value)

    # Create a list to store the names and tags of the attributes with varying values
    varying_attributes = [attribute for attribute, values in attribute_values.items() if len(values) > 1]

    return varying_attributes

def get_dicom_from_png(png_path,dataset_dir):
    # Extract parts from the PNG path
    parts = png_path.split('/')
    
    # Extract relevant parts for the DICOM path
    # Assuming the parts are in specific positions in the path
    pid = parts[-1].split('_')[0]
    date = parts[-1].split('_')[1]
    ori = parts[-1].split('_')[2]
    bp = parts[-1].split('_')[3]
    md = parts[-1].split('_')[4]
    sd = parts[-1].split('.png')[0].split('_')[-3]+'_'+parts[-1].split('.png')[0].split('_')[-2]+'_'+parts[-1].split('.png')[0].split('_')[-1]

    # Construct the DICOM file path
    # The base path might differ, adjust it according to your directory structure
 
    dicom_file_path = os.path.join(dataset_dir, pid, date, ori, bp,
                                   md, sd, "*.dcm")

    # Find the DICOM files using glob
    dicom_files = glob.glob(dicom_file_path)

    # Read the first DICOM file
    if dicom_files:
        ds = pydicom.dcmread(dicom_files[0])
        return os.path.dirname(dicom_files[0]) ,ds
    else:
        return None


def folders_with_multiple_trigger_times(root_dir):
    
    folder_paths = [x for x in glob.glob(root_dir+'/*/*/*/*/*/*') if os.path.isdir(x)]
    folders_with_multiple_triggers = []

    for folder in folder_paths:
        trigger_times = set()
        
        for filename in os.listdir(folder):
            if filename.endswith('.dcm'):
                full_path = os.path.join(folder, filename)
                
                try:
                    dicom_file = pydicom.dcmread(full_path)
                    trigger_time = getattr(dicom_file, 'TriggerTime', None)

                    if trigger_time is not None:
                        trigger_times.add(trigger_time)
                except:
                    print(f"Error reading {full_path}. Skipping.")

        if len(trigger_times) > 1:
            folders_with_multiple_triggers.append(folder)

    return folders_with_multiple_triggers

def rename_and_merge_folders(folders, old_name, new_name):
    processed = []
    for folder in folders:
        path_components = folder.split(os.sep)
        index = path_components.index(old_name)
        path_components = path_components[:index+1]
        old_folder_path = os.sep.join(path_components)
        if old_folder_path in processed:
            continue
        
        for i, component in enumerate(path_components):
            if component == old_name:
                path_components[i] = new_name
        new_folder_path = os.sep.join(path_components)

        if not os.path.exists(new_folder_path):
            os.rename(old_folder_path , new_folder_path)
        else:
            os.makedirs(new_folder_path,exist_ok=True)
            for file_name in os.listdir(old_folder_path):
                shutil.move(os.path.join(old_folder_path , file_name), new_folder_path)
        processed.append(old_folder_path)
        