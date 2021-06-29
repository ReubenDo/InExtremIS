import torch
from tqdm import tqdm 
import os
import argparse
import nibabel as nib
import pandas as pd
from geodesics import generate_geodesics

PHASES = ["training", "validation"]

def main():
    opt = parsing_data()
        
    # FOLDERS
    fold_dir = opt.output_folder
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
        
    # SPLIT
    assert os.path.isfile(opt.dataset_split), print("[ERROR] Invalid split file")
    df_split = pd.read_csv(opt.dataset_split,header =None)
    list_file = dict()
    for split in PHASES:
        list_file[split] = df_split[df_split[1].isin([split])][0].tolist()

    mod_ext = "_T2.nii.gz"
    grad_ext = f"_{opt.img_gradient_postfix}.nii.gz"
    extreme_ext = f"_{opt.label_postfix}.nii.gz"
    paths_dict = {split:[] for split in PHASES}

    print(f"Using the Euclidean distance: {opt.with_euclidean}")
    for split in PHASES:
        score = []
        for subject in tqdm(list_file[split]):
            subject_data = dict()

            img_path = os.path.join(opt.path_data,subject+mod_ext)
            img_grad_path = os.path.join(opt.path_extremes,subject+grad_ext)
            lab_path = os.path.join(opt.path_extremes,subject+extreme_ext)
            output_path = os.path.join(opt.output_folder,subject+'_PartLabel.nii.gz')

            if os.path.exists(img_path) and os.path.exists(lab_path) and os.path.exists(img_grad_path):
                extreme = nib.load(lab_path)
                affine = extreme.affine
                extreme_data = torch.from_numpy(extreme.get_fdata())

                grad_data = torch.from_numpy(nib.load(img_grad_path).get_fdata())

                geodesics = generate_geodesics(
                    extreme=extreme_data,
                    img_gradient=grad_data, 
                    prob=None,
                    with_prob=False,
                    with_euclidean=opt.with_euclidean).numpy().squeeze()

                nib.Nifti1Image(geodesics,affine).to_filename(output_path)

    
def parsing_data():
    parser = argparse.ArgumentParser(
        description="Script to generate (non-deep) geodesics using extreme points")

    parser.add_argument("--output_folder",
                    type=str,
                    default="geodesics_folder",
                    help="Path to the model directory")

    parser.add_argument("--dataset_split",
                    type=str,
                    default="splits/split_inextremis_budget1.csv",
                    help="Path to split file")

    parser.add_argument("--path_data",
                    type=str,
                    default="../data/VS_MICCAI21/T2/",
                    help="Path to the T2 scans")

    parser.add_argument("--path_extremes",
                    type=str,
                    default="../data/VS_MICCAI21/extremes_manual/",
                    help="Path to the extreme points")

    parser.add_argument("--label_postfix",
                    type=str,
                    default="Extremes_man",
                    help="Postfix of the Labels points")

    parser.add_argument("--img_gradient_postfix",
                    type=str,
                    default="Sobel_man",
                    help="Postfix of the gradient images")

    parser.add_argument("--with_euclidean", 
                    action="store_true", 
                    help="Add Euclidean distance")

    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    main()
