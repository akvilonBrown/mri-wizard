import dataloader_nmr
import numpy as np
import torch
import nibabel as nib

from dataloader_nmr import Compose, MoveAxis, Flip, RandomRotate90, ShiftScaleRotate, AddGaussianNoise, AddSpuriousNoise, MultiDataSet
from torch.utils.data import DataLoader

import models
import torch
import os
from pathlib import Path
import trainer
import inference
import time
import random
import pandas as pd
import matplotlib.pyplot as plt

import json
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# python train.py data.normalization="mean"
# A logger for this file
log = logging.getLogger(__name__)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    
    batch_size=cfg.train.batch_size #64
    random_seed = cfg.general.random_seed #42  
    set_seed(random_seed)
    pre = cfg.data.pre  #'/filer-5/user/plutenko/nmr_data/Testsets_Roeder/'
    cwd = os.getcwd()
    
    # getting current directroty two levels up since Hydre makes the current directory deeper
    cwd = hydra.utils.get_original_cwd()
    # handling relative path
    if pre.startswith(('.', '..')):
        pre = os.path.join(cwd, pre)
    log.info(f"Path to data: {pre}") 
    
    data_config_file = os.path.join(cwd, cfg.data.config_file)
    normalization = cfg.data.normalization
    frame_depth = cfg.train.input_depth
    input_dim = frame_depth*2+1 #number of frames left and right from the central frame + central frame
    #log.info(f"series: {cfg.series}")
    log.info(f"batch_size: {batch_size}")
    log.info(f"Data normalization mode: {normalization}")
    log.info(f"Input dimension: {input_dim}")

    with open(data_config_file) as json_file:
        datastr = json.load(json_file)
        log.info(datastr) 

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        log.error("CUDA unavailable")

    stest, ttest = dataloader_nmr.get_dataset_ext(prepath = pre, datastr = datastr, split = "test", normalization = normalization)
    log.info(f"stest type(stest){type(stest)}, stest.shape {stest.shape}, stest.dtype {stest.dtype}, stest.min() {stest.min()}, stest.max() {stest.max()}, stest.mean() {stest.mean()}")           
    log.info(f"ttest, type(ttest){type(ttest)}, ttest.shape {ttest.shape}, ttest.dtype {ttest.dtype}, ttest.min() {ttest.min()}, ttest.max() {ttest.max()}, np.unique(ttest) {np.unique(ttest)}")
    
    model_restored = models.UNet3(n_channels=input_dim, n_classes=5, bilinear=False).to(device)
    model_name = cfg.inference.model_name 
    if model_name.startswith(('.', '..')):
        model_name = os.path.join(cwd, model_name)
    log.info(model_name)
    model_weights = torch.load(model_name, map_location = device)    
    model_restored.load_state_dict(model_weights)

    transforms_test = Compose([
        MoveAxis(transform_target = False),
    ])
        
    
    #strain, ttrain, sval, tval, stest, ttest
    dataset_test = MultiDataSet(inputs=stest,
                                            targets=ttest,
                                            transform=transforms_test,
                                            depth = frame_depth) 
    
    dataloader_test = DataLoader(dataset=dataset_test,
                                     batch_size=batch_size,
                                     shuffle=False)
    
    
    log.info(f"len(dataloader_test): {len(dataloader_test)}")

    predictions = inference.predict_batch(model_restored, dataloader_test, device, )

    ni_img = nib.Nifti1Image(predictions, affine=np.eye(4))
    nib.save(ni_img, cfg.data.output_nii)
    
    iou = inference.iou_score_total(ttest, predictions)
    dice = inference.dice_score_total(ttest, predictions)
    metrics = iou.join(dice)
    log.info(metrics)
    metrics.to_csv(cfg.data.metrics_file, index = True, header=True)
    
 
if __name__ == '__main__': 
    main()
