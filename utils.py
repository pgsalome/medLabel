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
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

modality_considered = ["PT","CT","MR","DWI","PER","CB"]

#### string manip ####

def handle_none(value):
    if value is None:
        return 'NS'  # return an empty string if the value is None
    else:
        return value.replace("_","")
    

def split_string(s, phrases_to_keep= ['obere Extremitäten','DIFFUSION TENSOR',"PROJECTION IMAGE","MR gesamte WirbelsÃ¤ul","gesamte WirbelsÃ¤ul"]):

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

def remove_special_chars_and_numbers(s, remove_space=True):
    # This function uses a custom approach to iterate through the string
    # and build a new string according to the specified rules.
    
    new_s = []  # List to collect the characters that will form the new string
    i = 0  # Index to iterate through the original string
    while i < len(s):
        if s[i] == 'T' and i+1 < len(s) and s[i+1].isdigit():
            # Preserve "T" and the following numbers as a single unit
            new_s.append('T')
            i += 1
            while i < len(s) and s[i].isdigit():
                new_s.append(s[i])
                i += 1
        elif s[i].isalpha() or s[i] == '*':
            # Preserve alphabetic characters and asterisks
            new_s.append(s[i])
            i += 1
        else:
            # Handle other characters based on the remove_space flag
            if not remove_space:
                new_s.append(' ')
            i += 1  # Move to the next character

    return ''.join(new_s)


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





    
def remove_duplicates(label):
    parts = label.split('-')
    unique_parts = set(parts)
    return '-'.join(sorted(unique_parts))



####### initial curation scripts ####




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
    # background_value = np.min(edge_slices)
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
                            'constant', constant_values=0)

    return selected_slice
import matplotlib.cm as cm

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


def dicom_to_png_by_orientation(root_dir_or_dicomfolders, output_dir, desired_orientation="COR",target_size = (224, 224), plot_rgb=False):
    """
    Converts DICOM images to PNG by orientation, adjusting the aspect ratio based on pixel spacing and slice thickness.

    :param root_dir: The root directory where DICOM files are located.
    :param output_dir: The directory where PNG files will be saved.
    :param desired_orientation: The desired orientation ('COR', 'TRA', 'SAG') for the output image.
    :param pixel_spacing: The pixel spacing (x, y) from the DICOM metadata.
    :param slice_thickness: The slice thickness from the DICOM metadata.
    """
    
    failed = []
    is_4d = []
    if not isinstance(root_dir_or_dicomfolders, list):
    
        ignore_list = ["CR","US","XA","CR","PHS","PX","DX","PHS","IO","DX","CSR",
                       "RTDOSE", "RTSTRUCT", "RTIMAGE", 
                       "RTRECORD", "RTPLAN", "PR", "SR", "SC"]
        dicom_folders = [x for x in glob.glob(root_dir_or_dicomfolders + '/*/*/*/*/*/*') if os.path.isdir(x) and x.split('/')[-2] not in ignore_list]
    else:
        dicom_folders = root_dir_or_dicomfolders
    for dicom_folder in dicom_folders:
        check_4d = False
        
        print(dicom_folder)
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
            continue
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
            nifty_or = get_nifti_orientation(nii_d[0])
            print(nifty_or)
        except:
            nifty_or = "NS"
        
        if len(image_data_init.shape) > 3:
            check_4d = True
            channels = image_data_init.shape[-1]
            if channels > 10:
                channels = 10
            is_4d.append(dicom_folder)
        else:
            check_4d = False
            channels = 1

        for c in range(channels):
            if not check_4d:
                image_data = image_data_init
            else:
                try:
                    image_data = image_data_init[:,:,:,c]
                except:
                    print("error"+dicom_folder)
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
                         selected_slice = image_data[image_data.shape[0]//2, :,: ]
                     elif orientation == "TRA":
                         selected_slice = image_data[:,image_data.shape[1]//2,:]
                     else:
                         selected_slice = image_data[:,:,image_data.shape[2]//2]
                         
                 elif desired_orientation == "TRA":
                     if orientation == "SAG":
                         selected_slice = image_data[:,image_data.shape[1]//2,:]        
                     elif orientation == "TRA":
                         selected_slice = image_data[:,image_data.shape[1]//2,:]
                     else:
                         selected_slice = image_data[:,image_data.shape[1]//2,:]          
                 elif desired_orientation == "SAG":
                     if orientation == "SAG":
                         selected_slice = image_data[:,:,image_data.shape[2]//2]         
                     elif orientation == "TRA":
                         selected_slice = image_data[image_data.shape[0]//2,:,:]
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
            
    return is_4d, failed



def get_slice(nifti_path,image_data, nifty_or, orientation, desired_orientation):
    selected_slice = None
    print(nifty_or)
    # rot = check_for_rotation(nifti_path)
    # if rot:
    #     print("rotation")
    if nifty_or == "LPI":
        if desired_orientation == "COR":
            if orientation == "SAG":
                selected_slice = image_data[image_data.shape[0]//2, :, :]
            elif orientation == "TRA":
                selected_slice = image_data[:, image_data.shape[1]//2, :]
            else:
                selected_slice = image_data[:, :, image_data.shape[2]//2]
        elif desired_orientation == "TRA":
            if orientation == "SAG":
                selected_slice = image_data[:, :, image_data.shape[2]//2]
            elif orientation == "TRA":
                selected_slice = image_data[:, :, image_data.shape[2]//2]
            else:
                selected_slice = image_data[:, image_data.shape[1]//2, :]
        elif desired_orientation == "SAG":
            if orientation == "SAG":
                selected_slice = image_data[:, image_data.shape[1]//2, :]
            elif orientation == "TRA":
                selected_slice = image_data[image_data.shape[0]//2, :, :]
            else:
                selected_slice = image_data[image_data.shape[0]//2, :, :]
    elif nifty_or == "LAI" :
        if desired_orientation == "COR":
            if orientation == "SAG":
                selected_slice = image_data[image_data.shape[0]//2, :,: ]
            elif orientation == "TRA":
                selected_slice = image_data[:, image_data.shape[1]//2, :]
            else:
                selected_slice = image_data[:, :, image_data.shape[2]//2]
        elif desired_orientation == "TRA":
            if orientation == "SAG":
                selected_slice = image_data[:, image_data.shape[1]//2, :]
            elif orientation == "TRA":
                selected_slice = image_data[:, :, image_data.shape[2]//2]
            else:
                selected_slice = image_data[:, image_data.shape[1]//2, :]
        elif desired_orientation == "SAG":
            if orientation == "SAG":
                selected_slice = image_data[:, :, image_data.shape[2]//2]
            elif orientation == "TRA":
                selected_slice = image_data[image_data.shape[0]//2, :, :]
            else:
                selected_slice = image_data[image_data.shape[0]//2, :, :]
    elif nifty_or == "RPI":
        if desired_orientation == "COR":
            if orientation == "SAG":
                selected_slice = image_data[:, image_data.shape[1]//2,: ]
            elif orientation == "TRA":
                selected_slice = image_data[ image_data.shape[0]//2,:, :]
            else:
                selected_slice = image_data[:, :, image_data.shape[2]//2]
        elif desired_orientation == "TRA":
            if orientation == "SAG":
                selected_slice = image_data[:, :, image_data.shape[2]//2]
            elif orientation == "TRA":
                selected_slice = image_data[:, :, image_data.shape[2]//2]
            else:
                selected_slice = image_data[:, image_data.shape[1]//2, :]
        elif desired_orientation == "SAG":
            if orientation == "SAG":
                selected_slice = image_data[image_data.shape[0]//2, :, :]
            elif orientation == "TRA":
                selected_slice = image_data[:, image_data.shape[1]//2, :]
            else:
                selected_slice = image_data[image_data.shape[0]//2, :, :]
    return selected_slice


def select_nifti_slice(d, rootdir, desired_orientation,orientation_overwrite = None, plot = False):
    try:
        dicom_folder = get_dicom_from_png(d, rootdir)[0]
    except:
        dcm_folder = get_dicom_from_png(remove_last_number(png),rootdir)[0]
    orientation = dicom_folder.split("/")[-4]
    if '3D' in orientation or 'NA' in orientation:
        orientation = "TRA"
    elif 'SAG' in orientation:
        orientation = 'SAG'
    elif 'COR' in orientation:
        orientation = 'COR'
    elif 'TRA' in orientation:
        orientation = 'TRA'
    if orientation_overwrite:
        orientation = orientation_overwrite
    nii_d = glob.glob(dicom_folder + '.nii.gz')
    img = nib.load(nii_d[0])
    image_data_init = img.get_fdata()
    if len(image_data_init.shape) > 3:
        image_data_init = image_data_init[:,:,:,-1]
    nifty_or = get_nifti_orientation(nii_d[0])
    print(nifty_or)
    image_data = image_data_init
    selected_slice = get_slice(nii_d[0],image_data, nifty_or, orientation, desired_orientation)
    pixel_spacing,slice_thickness =  get_pixel_spacing_and_slice_thickness_from_nifti(nii_d[0])
    if pixel_spacing is None or len(pixel_spacing) == 0 or slice_thickness is None:
        ds = pydicom.read_file(glob.glob(dicom_folder+'/*.dcm')[0])
        pixel_spacing,slice_thickness =  get_pixel_spacing_and_slice_thickness(ds)
        
    if orientation != desired_orientation:
        selected_slice = correct_aspect(selected_slice, pixel_spacing, slice_thickness)    

    plt.imshow(selected_slice, cmap='gray')
    plt.show()   


# Your existing function with modifications



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
    
def get_orientation_from_sitk(nii_path):
    # Read the image
    img = sitk.ReadImage(nii_path)
    
    # Get direction cosines
    direction = img.GetDirection()
    
    # The direction cosines matrix is flattened, so for a 3D image, it would be 9 elements
    # We reshape it into a 3x3 matrix for easier interpretation
    direction_matrix = np.array(direction).reshape((3, 3))
    
    # Infer orientation based on the direction cosines
    # This is a simplified method to determine if the image is closer to RAS or not
    orientation_labels = ['R', 'A', 'S']  # Assuming positive direction for simplicity
    
    def axis_orientation(axis_cosines):
        # Find the dominant direction for the axis
        major_axis_index = np.argmax(np.abs(axis_cosines))
        sign = np.sign(axis_cosines[major_axis_index])
        label = orientation_labels[major_axis_index]
        return label if sign > 0 else label.lower()  # Lowercase if negative direction
    
    # Determine the orientation for each axis
    orientation = ''.join([axis_orientation(direction_matrix[:, i]) for i in range(3)])
    
    # Example interpretation (simplified)
    # "RAS" would mean normal neurological orientation
    # "LAS" would imply a flipped x-axis, common in radiological views
    orientation_type = 'neurological' if orientation.startswith('R') else 'radiological'
    
    return orientation


def get_rgb_image(nii_d, orientation,desired_orientation,target_size):
    # Read the image using SimpleITK
    img = sitk.ReadImage(nii_d)
    
    # Convert the image to a numpy array
    image_data = sitk.GetArrayFromImage(img)
    if len(image_data.shape) == 5:
        image_data = image_data[0,:,:,:,:]
    else:
        image_data = image_data[:,:,:,:]
    try:
        nifty_or = get_nifti_orientation(nii_d)
    except:
        nifty_or = get_orientation_from_sitk(nii_d)       
    
    spacing = img.GetSpacing()  # Returns a tuple (x_spacing, y_spacing, z_spacing)
    pixel_spacing = (spacing[0], spacing[1])  # Pixel spacing in x and y
    slice_thickness = spacing[2]  # Slice thickness (z spacing)
    
    if nifty_or == "neurological":
        if desired_orientation == "COR":
            if orientation == "SAG":
                selected_slice = image_data[image_data.shape[0]//2, :, :,:]
            elif orientation == "TRA":
                selected_slice = image_data[:, image_data.shape[1]//2, :,:]
            else:
                selected_slice = image_data[:, :, image_data.shape[2]//2,:]
        elif desired_orientation == "TRA":
            if orientation == "SAG":
                selected_slice = image_data[:, :, image_data.shape[2]//2,:]
            elif orientation == "TRA":
                selected_slice = image_data[:, :, image_data.shape[2]//2,:]
            else:
                selected_slice = image_data[:, image_data.shape[1]//2, :,:]
        elif desired_orientation == "SAG":
            if orientation == "SAG":
                selected_slice = image_data[:, image_data.shape[1]//2, :,:]
            elif orientation == "TRA":
                selected_slice = image_data[image_data.shape[0]//2, :, :,:]
            else:
                selected_slice = image_data[image_data.shape[0]//2, :, :,:]
    else:
        if desired_orientation == "COR":
            if orientation == "SAG":
                selected_slice = image_data[image_data.shape[0]//2, :,: ,:]
            elif orientation == "TRA":
                selected_slice = image_data[:, image_data.shape[1]//2, :,:]
            else:
                selected_slice = image_data[:, :, image_data.shape[2]//2,:]
        elif desired_orientation == "TRA":
            if orientation == "SAG":
                selected_slice = image_data[:, image_data.shape[1]//2, :,:]
            elif orientation == "TRA":
                selected_slice = image_data[:, :, image_data.shape[2]//2,:]
            else:
                selected_slice = image_data[:, image_data.shape[1]//2, :,:]
        elif desired_orientation == "SAG":
            if orientation == "SAG":
                selected_slice = image_data[:, :, image_data.shape[2]//2,:]
            elif orientation == "TRA":
                selected_slice = image_data[image_data.shape[0]//2, :, :,:]
            else:
                selected_slice = image_data[image_data.shape[0]//2, :, :,:]
    if orientation != desired_orientation:
        selected_slice = correct_aspect(selected_slice, pixel_spacing, slice_thickness, desired_aspect='COR')
    #### missing resize
    pil_img = Image.fromarray(selected_slice.astype(np.uint8), mode='RGB')
    if orientation == "COR":
        pil_img = pil_img.rotate(180)
    pil_img_resized = pil_img.resize(target_size, Image.ANTIALIAS)
    return pil_img_resized, selected_slice

def dicom_to_png_by_orientation_single(dicom_folder,desired_orientation="COR",target_size = (224, 224),plot=False):
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

    pil_img  = None
    orientation = dicom_folder.split("/")[-4]
    modality = dicom_folder.split("/")[-2]
    ds = pydicom.read_file(glob.glob(dicom_folder+'/*.dcm')[0])
    # patient_position = getattr(ds, 'PatientPosition', "HFS")
    # print(patient_position)
    # Adjust for additional orientations like '3DCOR', '3DSAG', etc.
    if '3D' in orientation or 'NA' in orientation:
        orientation = "TRA"
    elif 'SAG' in orientation:
        orientation = 'SAG'
    elif 'COR' in orientation:
        orientation = 'COR'
    elif 'TRA' in orientation:
        orientation = 'TRA'       
    # if patient_position != "HFS" and desired_orientation == "TRA" and orientation == "SAG":
    #     orientation_overwrite = True
    #     orientation = "TRA"
    try:
        img = nib.load(nii_d[0])
        image_data = img.get_fdata()
        if len(image_data.shape)>3:
            image_data = image_data[:,:,:,-1]
    except:
        if is_rgb(nii_d[0])[0]:
            rgb = True
            pil_img,resized_slice = get_rgb_image(nii_d[0],orientation, desired_orientation,target_size)
           

    if not pil_img:
        pixel_spacing,slice_thickness =  get_pixel_spacing_and_slice_thickness_from_nifti(nii_d[0])
    
        if pixel_spacing is None or len(pixel_spacing) == 0 or slice_thickness is None:
            
            pixel_spacing,slice_thickness =  get_pixel_spacing_and_slice_thickness(ds)
      
        nifty_or = get_nifti_orientation(nii_d[0])

        selected_slice = get_slice(nii_d[0],image_data, nifty_or, orientation, desired_orientation)
        if "CT" not in modality and "PT" not in modality:
            selected_slice[selected_slice<0] = 0
        # Check if selected_slice is None or not 2D
        if selected_slice is None or len(selected_slice.shape) != 2:
            print(f"No valid slice selected for folder: {dicom_folder}")
            return
    
        # Correcting the aspect ratio
        if orientation != desired_orientation :
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
        if orientation == "SAG" and desired_orientation == "COR" and nifty_or == "LPI" :
            pil_img = Image.fromarray(resized_slice.astype(np.uint8), mode='L').rotate(90)    
        if modality == "PT":
       
        # Apply the jet colormap
            colormap_image = cm.jet(resized_slice / 255)  # Normalize the image to range [0, 1]
            colormap_image = np.uint8(colormap_image * 255)  # Convert to 8-bit format
            pil_img = Image.fromarray(colormap_image, mode='RGBA')
    
    if plot: 
        plt.imshow(pil_img, cmap='gray')  # 'gray' colormap for grayscale
        plt.axis('off')  # Hide the axes
        plt.show()
    return pil_img

def check_for_rotation(nifti_path):
    """
    Check if the NIfTI image contains rotation by examining its affine matrix.

    Parameters:
    - nifti_path: Path to the NIfTI file.

    Returns:
    - has_rotation: Boolean indicating whether there is rotation in the image.
    - rotation_info: String describing the rotation status.
    """
    # Load the NIfTI image
    image = nib.load(nifti_path)
    affine = image.affine

    # Extract the rotation/scaling part of the affine matrix (upper-left 3x3 submatrix)
    rotation_scaling_matrix = affine[:3, :3]

    # Check for non-zero off-diagonal elements, which indicate rotation
    off_diagonal_elements = np.array([rotation_scaling_matrix[i, j]
                                      for i in range(3) for j in range(3) if i != j])
    has_rotation = not np.allclose(off_diagonal_elements, 0)

    if has_rotation:
        return True
    else:
        return False

    return has_rotation, rotation_info
def get_nifti_orientation(nifti_path):
    # Load the NIFTI file
    # if check_for_rotation(nifti_path):

    image = nib.load(nifti_path)
    affine = image.affine

    # Extract the rotation part of the affine matrix (3x3 upper-left submatrix)
    rotation = affine[:3, :3]

    # Determine the direction of each voxel axis in RAS+ space
    # by finding the column with the largest absolute value for each row.
    # This approach assumes a RAS+ (Right-Anterior-Superior) orientation as default.
    axis_directions = np.argmax(np.abs(rotation), axis=1)

    # Initialize an empty list to hold the orientation labels
    orientations = []

    for i, axis in enumerate(axis_directions):
        # Check the sign of the direction vector component to determine if it's positive or negative
        # and append the corresponding label with + or - to indicate the direction.
        sign = np.sign(rotation[i, axis])
        if i == 0:  # x-axis
            orientations.append(('L' if sign > 0 else 'R'))  # Flipped due to Radiological view
        elif i == 1:  # y-axis
            orientations.append(('P' if sign > 0 else 'A'))
        else:  # z-axis
            orientations.append(('I' if sign > 0 else 'S'))

    # Join the orientation labels to form a string that represents the orientation
    orientation_str = ''.join(orientations)

    # Interpret the orientation based on the anatomical labels
    # The orientation string now directly maps to the voxel orientation in the file,
    # taking into account the provided FSLeyes orientation information.
    # This interpretation could be adjusted based on specific needs or conventions.
    
    # Here, instead of determining neurological vs. radiological, we return the orientation string directly,
    # as the original orientation determination might not fully apply with the detailed FSLeyes info.
    return orientation_str 

# def get_nifti_orientation(nifti_path):
#     # Load the NIFTI file
#     image = nib.load(nifti_path)
#     affine = image.affine

#     # Extract the rotation part of the affine matrix (3x3 upper-left submatrix)
#     rotation = affine[:3, :3]

#     # Determine the direction of each voxel axis in RAS space
#     # by finding the column with the largest absolute value for each row
#     axis_directions = np.argmax(np.abs(rotation), axis=1)

#     # Map the axis directions to anatomical labels
#     labels = ['R', 'A', 'S']  # Right, Anterior, Superior directions for positive axis increments
#     orientations = []

#     for i, axis in enumerate(axis_directions):
#         # Check the sign of the direction vector component to determine if it's positive or negative
#         # and append the corresponding label with + or - to indicate the direction
#         sign = np.sign(rotation[i, axis])
#         if axis == 0:  # x-axis
#             orientations.append(('R' if sign > 0 else 'L'))
#         elif axis == 1:  # y-axis
#             orientations.append(('A' if sign > 0 else 'P'))
#         else:  # z-axis
#             orientations.append(('S' if sign > 0 else 'I'))

#     # Join the orientation labels to form a string that represents the orientation
#     orientation_str = ''.join(orientations)

#     # Determine if the orientation is 'neurological' (RAS) or 'radiological' (LAS) based on the x-axis direction
#     # In the context of this function, we simplify the definition:
#     # 'neurological' if the x-axis is towards the Right (first letter is 'R'), 
#     # and 'radiological' if towards the Left (first letter is 'L')
#     orientation = 'neurological' if orientation_str.startswith('R') else 'radiological'

#     return orientation
    
def numeric_sort_key(filepath):
    base_name = os.path.basename(filepath)
    parts = base_name.split('.dcm')
    numeric_part = parts[0] + '.' + parts[1] if len(parts) > 1 else parts[0]
    return float(numeric_part)



    


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
        return os.path.dirname(dicom_files[0]) ,ds, dicom_files[0]
    else:
        return None

