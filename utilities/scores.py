import argparse
import os
from tqdm import tqdm
from medpy.metric.binary import dc, precision, hd95
import numpy as np
import nibabel as nib
import pandas as pd
from natsort import natsorted

opt = argparse.ArgumentParser(
    description="Computing scores")

opt.add_argument("--model_dir",
                type=str,
                help="Path to the model directory")
opt.add_argument("--path_data",
                type=str,
                default="../data/VS_MICCAI21/T2/",
                help="Path to the labels")

opt = opt.parse_args()

name_folder = os.path.basename(os.path.normpath(opt.model_dir))

df_split = pd.read_csv('splits/split_inextremis_budget1.csv',header =None)
list_patient = natsorted(df_split[df_split[1].isin(['inference'])][0].tolist())

list_dice = []
list_hd = []
list_precision = []

df_scores= {'name':[],'dice':[],'hd95':[],'precision':[]}
for patient in tqdm(list_patient):
    path_gt = os.path.join(opt.path_data, patient+"_Label.nii.gz")
    path_pred = os.path.join(opt.model_dir,'output_pred',f"{patient}_T2",f"{patient}_T2_seg.nii.gz")
    gt = nib.funcs.as_closest_canonical(nib.load(path_gt)).get_fdata().squeeze()
    pred = nib.funcs.as_closest_canonical(nib.load(path_pred)).get_fdata().squeeze()
    affine = nib.funcs.as_closest_canonical(nib.load(path_gt)).affine

    voxel_spacing = [abs(affine[k,k]) for k in range(3)]
    dice_score = dc(pred, gt)
    if np.sum(pred)>0:
        hd_score = 0.0
        hd_score = hd95(pred, gt, voxelspacing=voxel_spacing)
    precision_score = precision(pred, gt)

    list_dice.append(100*dice_score)
    list_hd.append(hd_score)
    list_precision.append(100*precision_score)

    df_scores['name'].append(patient)
    df_scores['dice'].append(dice_score)
    df_scores['hd95'].append(hd_score)
    df_scores['precision'].append(precision_score)

df_scores = pd.DataFrame(df_scores)
df_scores.to_csv(os.path.join(opt.model_dir, "results_full.csv"))


mean_dice = np.round(np.mean(list_dice),1) 
std_dice = np.round(np.std(list_dice),1)
mean_hd = np.round(np.mean(list_hd),1) 
std_hd = np.round(np.std(list_hd),1) 
mean_precision = np.round(np.mean(list_precision),1) 
std_precision = np.round(np.std(list_precision),1) 

print(name_folder)
print(f"{mean_dice} ({std_dice}) & {mean_hd} ({std_hd}) & {mean_precision} ({std_precision}) \\")

