#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:42:32 2023

@author: pgsalome
consol19 chordoma
conso20 analca
consol21 cinderela
console14 CLEOPATRA_GBM
"""
from utils import handle_files,rearrange_files,delete_empty_dirs,convert_and_label_4d_png_txt_pl,delete_small_directories
root_dir = '/media/e210/portable_hdd/data_unsorted/GLIOMA/all_pids
output_dir = root_dir.replace('/media/e210/portable_hdd/data_unsorted/',
                              '/media/e210/B731-E206/')

output_dir = rearrange_files(root_dir,output_dir)
file_found = delete_small_directories(output_dir,9)
folder_found = True
while folder_found:
    folder_found = delete_empty_dirs(output_dir)
handle_files(output_dir)
# output_dir = '/media/e210/HD7/Cinderella'
convert_and_label_4d_png_txt_pl(output_dir)
