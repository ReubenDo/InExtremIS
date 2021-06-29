#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from tqdm import tqdm

import pandas as pd 

import torch
from torch import nn

from monai.data import DataLoader, Dataset, NiftiSaver
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    NormalizeIntensityd,
    Orientationd,
    ToTensord,
)
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference

from network.unet2d5 import UNet2D5

# Define training and patches sampling parameters   
SPATIAL_SHAPE = (224,224,48)

NB_CLASSES = 2

# Number of worker
workers = 20

# Training parameters
val_eval_criterion_alpha = 0.95
train_loss_MA_alpha = 0.95
nb_patience = 10
patience_lr = 5
weight_decay = 1e-5

PHASES = ['training', 'validation', 'inference']

def infinite_iterable(i):
    while True:
        yield from i

def inference(paths_dict, model, transform_inference, device, opt):
    
    # Define transforms for data normalization and augmentation
    dataloaders = dict()
    subjects_dataset = dict()

    checkpoint_path = os.path.join(opt.model_dir,'models', './CP_{}.pth')
    checkpoint_path = checkpoint_path.format(opt.epoch_inf)
    assert os.path.isfile(checkpoint_path), 'no checkpoint found'
    print(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))

    model = model.to(device) 

    for phase in ['inference']:
        subjects_dataset[phase] = Dataset(paths_dict, transform=transform_inference)
        dataloaders[phase] = DataLoader(subjects_dataset[phase], batch_size=1, shuffle=False)


    model.eval()  # Set model to evaluate mode

    fold_name = 'output_pred'
    # Iterate over data
    with torch.no_grad():
        saver = NiftiSaver(output_dir=os.path.join(opt.model_dir,fold_name))
        for batch in tqdm(dataloaders['inference']):
            inputs =   batch['img'].to(device)

            pred = sliding_window_inference(inputs, opt.spatial_shape, 1, model, mode='gaussian')
            
            pred = pred.argmax(1, keepdim=True).detach()
            saver.save_batch(pred, batch["img_meta_dict"])


def main():
    opt = parsing_data()

    set_determinism(seed=2)

    if torch.cuda.is_available():
        print('[INFO] GPU available.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise Exception(
            "[INFO] No GPU found.")
        

    print("[INFO] Reading data")
    # PHASES
    split_path = os.path.join(opt.dataset_split)
    df_split = pd.read_csv(split_path,header =None)
    list_file = dict()
    for phase in PHASES: # list of patient name associated to each phase
        list_file[phase] = df_split[df_split[1].isin([phase])][0].tolist()

    # CREATING DICT FOR DATASET
    mod_ext = "_T2.nii.gz"
    paths_dict = {split:[] for split in PHASES}
    for split in PHASES:
        for subject in list_file[split]:
            subject_data = dict()
            if os.path.exists(os.path.join(opt.path_data,subject+mod_ext)):
                subject_data["img"] = os.path.join(opt.path_data,subject+mod_ext)
                paths_dict[split].append(subject_data)
        print(f"Nb patients in {split} data: {len(paths_dict[split])}")

    # Logging hyperparameters
    print("[INFO] Hyperparameters")
    print('Spatial shape: {}'.format(opt.spatial_shape))
    print(f"Inference on the {opt.phase} set")
     
    # PREPROCESSING
    all_keys = ["img"]
    test_transforms = Compose(
        (
            LoadNiftid(keys=all_keys),
            AddChanneld(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            NormalizeIntensityd(keys=all_keys),
            ToTensord(keys=all_keys)
            )
    )

    # MODEL
    norm_op_kwargs = {"eps": 1e-5, "affine": True}  
    net_nonlin = nn.LeakyReLU   
    net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}   
    
    model= UNet2D5(input_channels=1,   
                base_num_features=16,   
                num_classes=NB_CLASSES,     
                num_pool=4,   
                conv_op=nn.Conv3d,    
                norm_op=nn.InstanceNorm3d,    
                norm_op_kwargs=norm_op_kwargs,  
                nonlin=net_nonlin,  
                nonlin_kwargs=net_nonlin_kwargs).to(device)

    print("[INFO] Inference")
    inference(paths_dict[opt.phase], model, test_transforms, device, opt)


def parsing_data():
    parser = argparse.ArgumentParser(
        description='Performing inference')


    parser.add_argument('--model_dir',
                        type=str)

    parser.add_argument("--dataset_split",
                        type=str,
                        default="splits/split_inextremis_budget1.csv")

    parser.add_argument("--path_data",
                        type=str,
                        default="data/VS_MICCAI21/T2/")

    parser.add_argument('--phase',
                        type=str,
                        default='inference')

    parser.add_argument('--spatial_shape',
                    type=int,
                    nargs="+",
                    default=(224,224,48))

    parser.add_argument('--epoch_inf',
                    type=str,
                    default='best')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()



