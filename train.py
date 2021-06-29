#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn 

from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    SpatialPadd,
    NormalizeIntensityd,
    RandFlipd,
    RandSpatialCropd,
    Orientationd,
    ToTensord,
)

from utilities.losses import DC, DC_CE_Focal, PartialLoss
from utilities.utils import (
    create_logger, 
    poly_lr, 
    infinite_iterable)
from utilities.geodesics import generate_geodesics

from ScribbleDA.scribbleDALoss import CRFLoss 
from network.unet2d5 import UNet2D5


# Define training and patches sampling parameters   
NB_CLASSES = 2
PHASES = ["training", "validation"]
MAX_EPOCHS = 300

# Training parameters
weight_decay = 3e-5

def train(paths_dict, model, transformation, criterion, device, save_path, logger, opt):
    
    since = time.time()

    # Define transforms for data normalization and augmentation
    subjects_train = Dataset(
        paths_dict["training"], 
        transform=transformation["training"])

    subjects_val = Dataset(
        paths_dict["validation"], 
        transform=transformation["validation"])
    
    # Dataloaders
    dataloaders = dict()
    dataloaders["training"] = infinite_iterable(
        DataLoader(subjects_train, batch_size=opt.batch_size, num_workers=2, shuffle=True)
        )
    dataloaders["validation"] = infinite_iterable(
        DataLoader(subjects_val, batch_size=1, num_workers=2)
        )

    nb_batches = {
        "training": 30, # One image patch per epoch for the full dataset
        "validation": len(paths_dict["validation"])
        }

    # Training parameters are saved 
    df_path = os.path.join(opt.model_dir,"log.csv")
    if os.path.isfile(df_path): # If the training already started
        df = pd.read_csv(df_path, index_col=False)
        epoch = df.iloc[-1]["epoch"]
        best_epoch = df.iloc[-1]["best_epoch"]
        best_val = df.iloc[-1]["best_val"]
        initial_lr = df.iloc[-1]["lr"]
        model.load_state_dict(torch.load(save_path.format("best")))

    else: # If training from scratch
        columns=["epoch","best_epoch", "MA", "best_MA", "lr", "timeit"]
        df = pd.DataFrame(columns=columns)
        best_val = None
        best_epoch = 0
        epoch = 0
        initial_lr = opt.learning_rate


    # Optimisation policy mimicking nnUnet training policy
    optimizer = torch.optim.SGD(model.parameters(),  initial_lr, 
                weight_decay=weight_decay, momentum=0.99, nesterov=True)
                
    # CRF Loss initialisation
    crf_l = CRFLoss(alpha=opt.alpha, beta=opt.beta, is_da=False, use_norm=False)

    # Training loop
    continue_training = True
    while continue_training:
        epoch+=1
        logger.info("-" * 10)
        logger.info("Epoch {}/".format(epoch))
        logger.info
        for param_group in optimizer.param_groups:
            logger.info("Current learning rate is: {}".format(param_group["lr"]))
            
        # Each epoch has a training and validation phase
        for phase in PHASES:
            if phase == "training":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode 

            # Initializing the statistics
            running_loss = 0.0
            running_loss_reg = 0.0
            running_loss_seg = 0.0
            epoch_samples = 0
            running_time = 0.0

            # Iterate over data
            for _ in tqdm(range(nb_batches[phase])):
                batch = next(dataloaders[phase])
                inputs = batch["img"].to(device) # T2 images
                if opt.mode == "extreme_points":
                    extremes = batch["label"].to(device) # Extreme points
                    img_gradients = batch["img_gradient"].to(device) # Pre-Computed Sobel map
                else:
                    labels = batch["label"].to(device)
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "training"):
                    if phase=="training": # Random patch predictions
                        outputs = model(inputs)
                    else:  # if validation, Inference on the full image
                        outputs = sliding_window_inference(
                            inputs=inputs,
                            roi_size=opt.spatial_shape,
                            sw_batch_size=1,
                            predictor=model,
                            mode="gaussian",
                        )

                    if opt.mode == "extreme_points": # Generate geodesics
                        init_time_geodesics = time.time()
                        geodesics = []
                        nb_target = outputs.shape[0]
                        for i in range(nb_target):
                            geodesics_i = generate_geodesics(
                                extreme=extremes[i,...], 
                                img_gradient=img_gradients[i,...], 
                                prob=outputs[i,...], 
                                with_prob=opt.with_prob, 
                                with_euclidean=opt.with_euclidean
                            )
                            geodesics.append(geodesics_i.to(device))
                        labels = torch.cat(geodesics,0)
                        time_geodesics = time.time() - init_time_geodesics
                    else:
                        time_geodesics = 0.   

                    # Segmentation loss
                    loss_seg = criterion(outputs, labels, phase) 

                    # CRF regularisation (training only)
                    if (opt.beta>0 or opt.alpha>0) and phase == "training" and opt.mode == "extreme_points":
                        reg = opt.weight_crf/np.prod(opt.spatial_shape)*crf_l(inputs, outputs)
                        loss =  loss_seg  + reg
                    else:
                        reg = 0.0
                        loss =  loss_seg

                    if phase == "training":
                        loss.backward()
                        optimizer.step()
                
                # Iteration statistics
                epoch_samples += 1
                running_loss += loss.item()
                running_loss_reg += reg
                running_loss_seg += loss_seg 
                running_time += time_geodesics

            # Epoch statistcs
            epoch_loss = running_loss / epoch_samples
            epoch_loss_reg = running_loss_reg / epoch_samples
            epoch_loss_seg = running_loss_seg / epoch_samples
            if phase == "training":
                epoch_time = running_time / epoch_samples
           
            logger.info("{}  Loss Reg: {:.4f}".format(
                phase, epoch_loss_reg))
            logger.info("{}  Loss Seg: {:.4f}".format(
                phase, epoch_loss_seg))
            if phase == "training":
                logger.info("{}  Time Geodesics: {:.4f}".format(
                    phase, epoch_time))
                
            # Saving best model on the validation set
            if phase == "validation":
                if best_val is None: # first iteration
                    best_val = epoch_loss
                    torch.save(model.state_dict(), save_path.format("best"))

                if epoch_loss <= best_val:
                    best_val = epoch_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path.format("best"))

                df = df.append(
                    {"epoch":epoch,
                    "best_epoch":best_epoch,
                    "best_val":best_val,  
                    "lr":param_group["lr"],
                    "timeit":epoch_time}, 
                    ignore_index=True)
                df.to_csv(df_path, index=False)

                optimizer.param_groups[0]["lr"] = poly_lr(epoch, MAX_EPOCHS, opt.learning_rate, 0.9)

            # Early stopping performed when full annotations are used (training set may be small)
            if opt.mode == "full_annotations" and epoch-best_epoch>70: 
                torch.save(model.state_dict(), save_path.format("final"))
                continue_training=False   
        
            if epoch == MAX_EPOCHS:
                torch.save(model.state_dict(), save_path.format("final"))
                continue_training=False
    
    time_elapsed = time.time() - since
    logger.info("[INFO] Training completed in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info(f"[INFO] Best validation epoch is {best_epoch}")


def main():
    set_determinism(seed=2)

    opt = parsing_data()
        
    # FOLDERS
    fold_dir = opt.model_dir
    fold_dir_model = os.path.join(fold_dir,"models")
    if not os.path.exists(fold_dir_model):
        os.makedirs(fold_dir_model)
    save_path = os.path.join(fold_dir_model,"./CP_{}.pth")

    if opt.path_labels is None:
        opt.path_labels = opt.path_data

    logger = create_logger(fold_dir)
    logger.info("[INFO] Hyperparameters")
    logger.info(f"Alpha: {opt.alpha}")
    logger.info(f"Beta: {opt.beta}")
    logger.info(f"Weight Reg: {opt.weight_crf}")
    logger.info(f"Batch size: {opt.batch_size}")
    logger.info(f"Spatial shape: {opt.spatial_shape}")
    logger.info(f"Initial lr: {opt.learning_rate}")
    logger.info(f"Postfix img gradients: {opt.img_gradient_postfix}")
    logger.info(f"Postfix labels: {opt.label_postfix}")
    logger.info(f"With euclidean: {opt.with_euclidean}")
    logger.info(f"With probs: {opt.with_prob}")

    # GPU CHECKING
    if torch.cuda.is_available():
        logger.info("[INFO] GPU available.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise logger.error(
            "[INFO] No GPU found")

    # SPLIT
    assert os.path.isfile(opt.dataset_split), logger.error("[ERROR] Invalid split")
    df_split = pd.read_csv(opt.dataset_split,header =None)
    list_file = dict()
    for split in PHASES:
        list_file[split] = df_split[df_split[1].isin([split])][0].tolist()


    # CREATING DICT FOR CACHEDATASET
    mod_ext = "_T2.nii.gz"
    grad_ext = f"_{opt.img_gradient_postfix}.nii.gz"
    extreme_ext = f"_{opt.label_postfix}.nii.gz"
    paths_dict = {split:[] for split in PHASES}

    for split in PHASES:
        for subject in list_file[split]:
            subject_data = dict()

            img_path = os.path.join(opt.path_data,subject+mod_ext)
            img_grad_path = os.path.join(opt.path_labels,subject+grad_ext)
            lab_path = os.path.join(opt.path_labels,subject+extreme_ext)

            if os.path.exists(img_path) and os.path.exists(lab_path):
                subject_data["img"] = img_path
                subject_data["label"] = lab_path
                
                if opt.mode == "extreme_points":
                    if os.path.exists(img_grad_path):
                        subject_data["img_gradient"] = img_grad_path
                        paths_dict[split].append(subject_data)
                else:
                     paths_dict[split].append(subject_data)
                
        logger.info(f"Nb patients in {split} data: {len(paths_dict[split])}")
            

    # PREPROCESSING
    transforms = dict()
    all_keys = ["img", "label"]
    if opt.mode == "extreme_points":
        all_keys.append("img_gradient")

    transforms_training = (
        LoadNiftid(keys=all_keys),
        AddChanneld(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        NormalizeIntensityd(keys=["img"]),
        SpatialPadd(keys=all_keys, spatial_size=opt.spatial_shape),
        RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
        RandSpatialCropd(keys=all_keys, roi_size=opt.spatial_shape, random_center=True, random_size=False),
        ToTensord(keys=all_keys),
        )   
    transforms["training"] = Compose(transforms_training)   

    transforms_validation = (
        LoadNiftid(keys=all_keys),
        AddChanneld(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        NormalizeIntensityd(keys=["img"]),
        SpatialPadd(keys=all_keys, spatial_size=opt.spatial_shape),
        ToTensord(keys=all_keys)
        )   
    transforms["validation"] = Compose(transforms_validation) 
 
    # MODEL
    logger.info("[INFO] Building model")   
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

  
    logger.info("[INFO] Training")
    if opt.mode == "full_annotations":
        dice = DC(NB_CLASSES)
        criterion = lambda pred, grnd, phase: dice(pred, grnd)

    elif opt.mode == "extreme_points" or opt.mode == "geodesics":
        dice_ce_focal = DC_CE_Focal(NB_CLASSES)
        criterion = PartialLoss(dice_ce_focal)
    
    train(paths_dict, 
        model, 
        transforms, 
        criterion, 
        device, 
        save_path,
        logger,
        opt)

def parsing_data():
    parser = argparse.ArgumentParser(
        description="Script to train the models using extreme points as supervision")

    parser.add_argument("--model_dir",
                    type=str,
                    help="Path to the model directory")

    parser.add_argument("--mode",
                    type=str,
                    help="Choice of the supervision mode",
                    choices=["full_annotations", "extreme_points", "geodesics"],
                    default="extreme_points")

    parser.add_argument("--weight_crf",
                    type=float,
                    default=0.1)

    parser.add_argument("--alpha",
                    type=float,
                    default=15)

    parser.add_argument("--beta",
                    type=float,
                    default=0.05)

    parser.add_argument("--batch_size",
                    type=int,
                    default=6,
                    help="Size of the batch size (default: 6)")

    parser.add_argument("--dataset_split",
                    type=str,
                    default="splits/split_inextremis_budget1.csv",
                    help="Path to split file")

    parser.add_argument("--path_data",
                    type=str,
                    default="data/VS_MICCAI21/T2/",
                    help="Path to the T2 scans")

    parser.add_argument("--path_labels",
                    type=str,
                    default=None,
                    help="Path to the extreme points")

    parser.add_argument("--learning_rate",
                    type=float,
                    default=1e-2,
                    help="Initial learning rate")

    parser.add_argument("--label_postfix",
                    type=str,
                    default="",
                    help="Postfix of the Labels points")

    parser.add_argument("--img_gradient_postfix",
                    type=str,
                    default="",
                    help="Postfix of the gradient images")

    parser.add_argument("--spatial_shape",
                    type=int,
                    nargs="+",
                    default=(224,224,48),
                    help="Size of the window patch")

    parser.add_argument("--with_prob", 
                    action="store_true", 
                    help="Add Deep probabilities")

    parser.add_argument("--with_euclidean", 
                    action="store_true", 
                    help="Add Euclidean distance")

    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    main()



