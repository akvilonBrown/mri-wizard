import dataloader_nmr
import numpy as np
import torch
import nibabel as nib

from dataloader_nmr import Compose, MoveAxis, TwoDimDataSet
from torch.utils.data import DataLoader

import models
import torch
import os
import inference
import random

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# python single_predict.py data.normalization="mean"
# A logger for this file
log = logging.getLogger(__name__)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def load_single_source(source_p, normalization, transpose):
    img_nii = nib.load(source_p)
    #img_nii = nib.as_closest_canonical(img_nii)
    img_np = img_nii.get_fdata(dtype=np.float32)
    log.info(img_np.shape)
    if transpose:
        img_np = np.moveaxis(img_np, 0, 1)
    img_np_norm = dataloader_nmr.normalize_intensity(img_np, normalization, percentile=0.8)
    if(len(img_np_norm.shape)==3): #source shape shoud be [x, 180, 180, 1]
        img_np_norm = np.expand_dims(img_np_norm, -1)
    assert len(img_np_norm.shape) == 4 , "Incorrect source array shape"
    
    return img_np_norm, img_nii.header    

@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    
    batch_size=cfg.train.batch_size #64
    random_seed = cfg.general.random_seed #42  
    set_seed(random_seed)
    cwd = hydra.utils.get_original_cwd()
 
  
    normalization = cfg.data.normalization
    frame_depth = cfg.train.input_depth
    input_dim = frame_depth*2+1 #number of frames left and right from the central frame + central frame
    log.info(f"batch_size: {batch_size}")
    log.info(f"Data normalization mode: {normalization}")
    log.info(f"Input dimension: {input_dim}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
         log.error("CUDA unavailable")
         device = torch.device('cpu')  
    
    model_restored = models.UNet3(n_channels=input_dim, n_classes=5, bilinear=False).to(device)
    model_name = cfg.inference.model_name
    if model_name.startswith(('.', '..')):
        model_name = os.path.join(cwd, model_name)
    log.info(model_name)
    model_weights = torch.load(model_name, map_location = device)    
    model_restored.load_state_dict(model_weights)

    source_path = cfg.inference.pickup_folder
    target_path = cfg.inference.save_folder
    if source_path.startswith(('.', '..')):
        source_path = os.path.join(cwd, source_path)
    if target_path.startswith(('.', '..')):
        target_path = os.path.join(cwd, target_path)        

    log.info(f"Prediction from the folder {source_path}") 
    log.info(f"Target folder {target_path}")

    transforms_test = Compose([
                MoveAxis(transform_target = False),
        ])      

    files = os.listdir(source_path)
    for file in files:
        print(file)
        load_path = os.path.join(source_path, file)
        img_np_norm, header = load_single_source(load_path, normalization, transpose = 0)
        log.info(f"{img_np_norm.shape = }, {img_np_norm.dtype = }")
    
        dataset_test = MultiDataSet(inputs=img_np_norm,
                                                targets=img_np_norm,
                                                transform=transforms_test,
                                                depth = frame_depth)
        dataloader_test = DataLoader(dataset=dataset_test,
                                         batch_size=batch_size,
                                         shuffle=False)
        log.info(f"{len(dataloader_test) = }")    
        predictions = inference.predict_batch(model_restored, dataloader_test, device, )    
        ni_img = nib.Nifti1Image(predictions, affine=np.eye(4))
        ni_img.header['pixdim'] = header['pixdim']
        save_path = os.path.join(target_path, file)
        nib.save(ni_img, save_path)
    

if __name__ == '__main__': 
    main()
