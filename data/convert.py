# Adapted from: https://github.com/KCL-BMEIS/VS_Seg/blob/master/preprocessing/TCIA_data_convert_into_convenient_folder_structure.py

#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob
from natsort import natsorted
import pydicom
import dicom2nifti
import re
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert the T2 scans from the TCIA dataset into the Nifti format')
parser.add_argument('--input', type=str, help='(string) path to TCIA dataset, in "Descriptive Directory Name" format, for example /home/user/.../manifest-1614264588831/Vestibular-Schwannoma-SEG')
parser.add_argument('--output', type=str, help='(string) path to output folder')
args = parser.parse_args()

input_path = args.input 
output_path = args.output 

if not os.path.isdir(output_path):
    os.makedirs(output_path, exist_ok=True)

cases = natsorted(glob(os.path.join(input_path, '*')))

for case in tqdm(cases):
    folders = glob(case+'/*/*')
    
    MRs = [] 
    MRs_paths = []
    
    for folder in folders:
        first_file = glob(folder+"/*")[0]
        dd = pydicom.read_file(first_file)

        if dd['Modality'].value == 'MR':
            MRs.append(dd)
            MRs_paths.append(first_file)
            
    found = False
    file_paths = None
    # sort for T2
    for MR, path in zip(MRs, MRs_paths):
        if "t2_" in MR['SeriesDescription'].value:
            MR_T2 = MR
            found = True
            file_paths = path
        else:
            raise Exception

    # write files into new folder structure
    p = re.compile(r'VS-SEG-(\d+)')
    case_idx = int(p.findall(case)[0])

    new_T2_path = os.path.join(output_path, 'vs_gk_' + str(case_idx) +'_T2.nii.gz')
    
    old_T2_folder = os.path.dirname(file_paths)
    
    dicom2nifti.dicom_series_to_nifti(old_T2_folder, new_T2_path, reorient_nifti=False)
        