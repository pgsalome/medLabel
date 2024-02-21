
import subprocess as sp
import os
from nipype.interfaces.dcm2nii import Dcm2niix
import glob


class DicomConverters():

    def __init__(self, dicom, ext='.nii.gz', clean=False):
        #print ('\nStarting the DICOM conversion of {0}...'.format(dicom.split('/')[-1]))
    
        self.dicom_folder = dicom
        self.filename = dicom.split('/')[-1]
        self.ext = ext
        self.outpath = os.path.dirname(dicom)

        
       
    # def dcm2niix_converter(self, compress=True,outpath=None):
        
    #     if not outpath:
    #         outpath = self.outpath
        
    #     if compress:
    #         cmd = ("dcm2niix -b n -m y -o \"{0}\" -f \"{1}\" -z y \"{2}\" ".format(outpath, self.filename,
    #                                                         self.dicom_folder))
    #     else:
    #         cmd = ("dcm2niix -b n -m y -o \"{0}\" -f \"{1}\" -z no \"{2}\" ".format(outpath, self.filename,
    #                                                    self.dicom_folder))
    #     #print(cmd)
    #     sp.check_output(cmd, shell=True)

    def dcm2niix_converter(self, compress=True, outpath=None):
        if not outpath:
            outpath = self.outpath

        converter = Dcm2niix()
        converter.inputs.source_dir = self.dicom_folder
        converter.inputs.output_dir = outpath
        converter.inputs.bids_format = False
        converter.inputs.merge_imgs = True
        #converter.inputs.ignore_deriv=True
        converter.inputs.compress = 'y' if compress else 'n'
        converter.inputs.single_file = True
        converter.inputs.out_filename = self.filename
        converter.terminal_output = 'allatonce'  # Suppress output
        result = converter.run()   
        output = result.runtime.stdout + result.runtime.stderr
        if "gantry tilt" in output.lower() or "missing images?" in output.lower() or "incompatible with nifti" in output.lower() :

            converted_files = glob.glob(os.path.join(outpath, self.filename + '*.ni*'))
            for file in converted_files:
                os.remove(file)
            return self.dicom_folder
        # Get the list of all files in the output directory with the specified filename
        converted_files = glob.glob(os.path.join(outpath, self.filename + '*.ni*'))
        if not converted_files:
            return self.dicom_folder
        else:
            self.rename_largest_file(self.check_similarsd(converted_files)) 
            return None
            
        #all_files = glob.glob(outpath)
        
    def rename_largest_file(self, file_list):
        # Get the file with the maximum size in the list
        largest_file = max(file_list, key=os.path.getsize)
        # Delete all other files
        for file in file_list:
            if file != largest_file:
                os.remove(file)

        # Rename this file
        new_name = os.path.join(self.outpath, self.filename + self.ext)
        os.rename(largest_file, new_name)

    def check_similarsd(self,converted_files):
        dcm_folders = [x for x in glob.glob(self.outpath+'/*') if os.path.isdir(x)]
        for file in converted_files:
            for dcm_folder in dcm_folders:
                if file.split('.nii')[0] == dcm_folder and dcm_folder != self.dicom_folder:
                    converted_files.remove(file)
        return converted_files
     