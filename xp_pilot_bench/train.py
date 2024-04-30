import dataloader_nmr
import numpy as np
import torch
import nibabel as nib

from dataloader_nmr import Compose, MoveAxis, Flip, RandomRotate90, ShiftScaleRotate, AddGaussianNoise, AddSpuriousNoise, TwoDimDataSet
from torch.utils.data import DataLoader

import models
import torch
import os
import trainer
import inference
import random
import pandas as pd
import matplotlib.pyplot as plt

import json
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# python train.py -m +series=0,1,2,3,4
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
    pre = cfg.data.pre 

    # getting current directroty two levels up since Hydre makes the current directory deeper
    cwd = hydra.utils.get_original_cwd()

    # handling relative path
    if pre.startswith(('.', '..')):
        pre = os.path.join(cwd, pre)
    log.info(f"Path to data: {pre}")
    
    folder = cfg.data.saved_models_folder  #'saved_models'
    data_config_file = os.path.join(cwd, cfg.data.config_file)
    normalization = cfg.data.normalization    
    log.info(f"Data normalization mode: {normalization}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        log.error("CUDA unavailable")

    
    model = models.UNet3(n_channels=1, n_classes=5, bilinear=False).to(device)

    with open(data_config_file) as json_file:
        datastr = json.load(json_file)        
        log.info(datastr)


    strain, ttrain =  dataloader_nmr.get_dataset_ext(prepath = pre, datastr = datastr, split = "train", normalization = normalization)
    sval, tval = dataloader_nmr.get_dataset_ext(prepath = pre, datastr = datastr, split = "validation", normalization = normalization)
    log.info(f"strain type(strain){type(strain)}, strain.shape {strain.shape}, strain.dtype {strain.dtype}, strain.min() {strain.min()}, strain.max() {strain.max()}, strain.mean() {strain.mean()}")           
    log.info(f"ttrain, type(ttrain){type(ttrain)}, ttrain.shape {ttrain.shape}, ttrain.dtype {ttrain.dtype}, ttrain.min() {ttrain.min()}, ttrain.max() {ttrain.max()}, np.unique(ttrain) {np.unique(ttrain)}")    
 
    log.info(f"sval type(sval){type(sval)}, sval.shape {sval.shape}, sval.dtype {sval.dtype}, sval.min() {sval.min()}, sval.max() {sval.max()}, sval.mean() {sval.mean()}")           
    log.info(f"tval, type(tval){type(tval)}, tval.shape {tval.shape}, tval.dtype {tval.dtype}, tval.min() {tval.min()}, tval.max() {tval.max()}, np.unique(tval) {np.unique(tval)}")

    # parameters for AddSpuriousNoise
    mmean = (strain.max()-strain.mean())*.2
    smean = strain.mean() + mmean
    
    var_limit=mmean/2
    clip_min = strain.mean()
    clip_max = strain.max()*0.5

    transforms_training = Compose([
        Flip(transform_target = True, p = 0.3),
        RandomRotate90(p = 0.3),
        ShiftScaleRotate(transform_target = True, scale_limit = 0.3, p=0.5),
        AddGaussianNoise(mean = 1, var_limit = 1,  p = 0.5 ),
        AddSpuriousNoise(
            mean=smean, 
            var_limit=var_limit,
            clip_min = clip_min,
            clip_max = clip_max,
            p = 0.3),
        MoveAxis(transform_target = False),
    ])
        
    transforms_validation = Compose([
        MoveAxis(transform_target = False),
    ])    
    
    dataset_training = TwoDimDataSet(inputs=strain,
                                            targets=ttrain,
                                            transform=transforms_training)
    
    dataset_validation = TwoDimDataSet(inputs=sval, 
                                            targets=tval,
                                            transform=transforms_validation)
    
    set_seed(random_seed)

    dataloader_training = DataLoader(dataset=dataset_training,
                                     batch_size=batch_size,
                                     shuffle=True)
    
    dataloader_validation = DataLoader(dataset=dataset_validation,
                                     batch_size=batch_size,
                                     shuffle=True)
    
    log.info(len(dataloader_training))
    log.info(len((dataloader_validation)))  
    log.info("Data prepared")
    

    
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    lr = cfg.train.lr #0.0002
    up_lr = cfg.train.up_lr #0.0008
    num_epochs = cfg.train.num_epochs #50 
    steps = len(dataset_training)//batch_size
    log.info("Steps: " + str(steps))
    s_up = steps * num_epochs//15  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_sheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=up_lr, step_size_up=s_up,mode="triangular2", cycle_momentum=False)

    trainer_object = trainer.Trainer_lr(model=model,
                      device=device,
                      criterion=criterion,
                      optimizer=optimizer,
                      training_DataLoader=dataloader_training,
                      validation_DataLoader=dataloader_validation,
                      lr_scheduler=lr_sheduler,
                      epochs=num_epochs,
                      epoch=0,
                      notebook=cfg.train.notebook,
                      best_model_dir = folder,
                      saving_milestone = cfg.train.saving_milestone )#5
    # start training
    training_losses, validation_losses, lr_rates = trainer_object.run_trainer()
    
    data = {'train_losses': training_losses, 
            'val_losses': validation_losses}
    df = pd.DataFrame(data)
    df.to_csv (cfg.train.losses_file, index = None, header=True)

    data = {'lr_rates': lr_rates}
    df = pd.DataFrame(data)
    df.to_csv(cfg.train.lr_rates_file, index = None, header=True) 
    log.info("Training completed")

    fig = inference.plot_training_v(training_losses, validation_losses, lr_rates, gaussian=True, sigma=1, figsize=(10, 4))
    plt.savefig(cfg.train.savefig_png)
    plt.savefig(cfg.train.savefig_pdf)  
    log.info("Plots saved") 

    model_restored = models.UNet3(n_channels=1, n_classes=5, bilinear=False).to(device)
    model_name = cfg.restore.model_name #'saved_models/best_model.pt'
    if model_name.startswith(('.', '..')):
        model_name = os.path.join(cwd, model_name)
    log.info(model_name)    
    model_weights = torch.load(model_name)    
    model_restored.load_state_dict(model_weights)      
    
    stest, ttest = dataloader_nmr.get_dataset_ext(prepath = pre, datastr = datastr, split = "test", normalization = normalization)
    log.info(f"stest type(stest){type(stest)}, stest.shape {stest.shape}, stest.dtype {stest.dtype}, stest.min() {stest.min()}, stest.max() {stest.max()}, stest.mean() {stest.mean()}")           
    log.info(f"ttest, type(ttest){type(ttest)}, ttest.shape {ttest.shape}, ttest.dtype {ttest.dtype}, ttest.min() {ttest.min()}, ttest.max() {ttest.max()}, np.unique(ttest) {np.unique(ttest)}")

    transforms_test = Compose([
        MoveAxis(transform_target = False),
    ])

    dataset_test = TwoDimDataSet(inputs=stest,
                                            targets=ttest,
                                            transform=transforms_test) 
    
    dataloader_test = DataLoader(dataset=dataset_test,
                                     batch_size=batch_size,
                                     shuffle=False)
    
    
    log.info(f"len(dataloader_test): {len(dataloader_test)}")

    predictions = inference.predict_batch(model, dataloader_test, device, )

    ni_img = nib.Nifti1Image(predictions, affine=np.eye(4))
    nib.save(ni_img, cfg.data.output_nii)
    
    iou = inference.iou_score_total(ttest, predictions)
    dice = inference.dice_score_total(ttest, predictions)

    metrics = iou.join(dice)
    log.info(metrics)
    metrics.to_csv(cfg.data.metrics_file, index = True, header=True)
if __name__ == '__main__': 
    main()


 


