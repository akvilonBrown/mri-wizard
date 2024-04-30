import dataloader_nmr
import numpy as np
import torch
import torch.backends.cudnn


from torch.utils.data import DataLoader
import models
import os
import trainer
import inference
import random
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from collections import namedtuple 
from sklearn.metrics import r2_score
from dataloader_nmr import Compose, ShiftScaleRotateCustom, MoveAxis

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# A logger for this file
log = logging.getLogger(__name__)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_set(dframe, datafolder, shape = (180, 180)):
    dset = np.zeros(((len(dframe),)+ shape), dtype = np.uint8)
    for i in range(len(dframe)):
        dset[i] = io.imread(os.path.join(datafolder, dframe.iloc[i, 0]))
    return np.expand_dims((dset > 0).astype(np.uint8), -1)    

@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    
    set_seed(cfg.general.random_seed)
    dfpath = cfg.data.dataframe_folder 
    datapath = cfg.data.dataset_folder 
    # getting current directroty two levels up since Hydre makes the current directory deeper
    cwd = hydra.utils.get_original_cwd()
    # handling relative path
    if dfpath.startswith(('.', '..')):
        dfpath = os.path.join(cwd, dfpath)
    log.info(f"Path to dataframe: {dfpath}")  
    if datapath.startswith(('.', '..')):
        datapath = os.path.join(cwd, datapath)
    log.info(f"Path to data: {datapath}")      
  
  
    num_workers = cfg.train.num_workers
    if num_workers == 0 or num_workers > len(os.sched_getaffinity(0)):
       num_workers = len(os.sched_getaffinity(0))
       log.info(f"num_workers was adjusted to the available cpu count. Count from config file: {cfg.train.num_workers}")

    log.info(f"num_workers: {num_workers}")   

    df_data = pd.read_csv(dfpath, index_col=[0])
    df_train = df_data[df_data.split == "train"]
    df_val = df_data[df_data.split == "val"]
    df_test = df_data[df_data.split == "test"]
    df_train.reset_index(inplace = True, drop=True)
    df_val.reset_index(inplace = True, drop=True)
    df_test.reset_index(inplace = True, drop=True)

    dtrain = load_set(df_train, datapath)
    dval = load_set(df_val, datapath)
    dtest = load_set(df_test, datapath)
    
    log.info(f"{dtrain.shape = }, {dval.shape = } , {dtest.shape = }")


 
    log.info("Data prepared")

    training_batch_size = cfg.train.batch_size
    validation_batch_size = 2 * training_batch_size

    transforms_training = Compose([
        ShiftScaleRotateCustom(transform_target = False, scale_limit = (-0.1, 0.7), rotate_limit=0,  p=0.5), # type: ignore        
        MoveAxis(transform_target = False)
        ])

    transforms_validation = Compose([    
        MoveAxis(transform_target = False)
        ])

    dataset_train = dataloader_nmr.RotationDataSet(dtrain, df_train, transforms_training, sine = True)
    dataset_val = dataloader_nmr.RotationDataSet(dval, df_val, transforms_validation, sine = True) 

    log.info(f"{len(dataset_train) = }, {len(dataset_val) = }")

    set_seed(cfg.general.random_seed)

    # dataloader training    
    dataloader_training = DataLoader(dataset=dataset_train,
                                     batch_size=training_batch_size,
                                     shuffle=True,
                                     num_workers=num_workers, 
                                     pin_memory=True)

    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_val,
                                       batch_size=validation_batch_size,
                                       shuffle=False,
                                       num_workers=num_workers, 
                                       pin_memory=True)                                    

    log.info(f"Training loader size: {len(dataloader_training)}")
    log.info(f"Validation loader size: {len(dataloader_validation)}") 

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
         log.error("CUDA unavailable")
         device = torch.device('cpu')

    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

    resnet18_4_config = ResNetConfig(block = models.BasicBlock,
                                   n_blocks = cfg.model.n_blocks,
                                   channels = cfg.model.channels)
 
    model = models.ResNetReg(
        config = resnet18_4_config, 
        input_dim = cfg.model.in_channels, 
        output_dim = cfg.model.outputs
         ).to(device)
    
    # optimizer
    if cfg.train.optimizer == "MSELoss":
        criterion = torch.nn.MSELoss()
    elif cfg.train.optimizer == "L1Loss":
        criterion = torch.nn.L1Loss()
    else: 
        criterion = torch.nn.MSELoss()

    log.info(f"Loss: {cfg.train.optimizer}")
    
    lr = cfg.train.lr #0.0002
    up_lr = cfg.train.up_lr #0.0008
    num_epochs = cfg.train.num_epochs #50 
    steps = len(dataset_train)//training_batch_size
    log.info("Steps: " + str(steps))
    s_up = steps * num_epochs//cfg.train.steps_divider #15  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_sheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=up_lr, step_size_up=s_up,mode="triangular2", cycle_momentum=False)    

    log.info("Model prepared")    

    trainer_object = trainer.Trainer_lr(
                     model=model,
                     device=device,
                     criterion=criterion,
                     optimizer=optimizer,
                     training_DataLoader=dataloader_training,
                     validation_DataLoader=dataloader_validation,
                     lr_scheduler=lr_sheduler,         
                     epochs=cfg.train.num_epochs,
                     epoch=0,
                     notebook=cfg.train.notebook,
                     best_model_dir = cfg.data.saved_models_folder, #"saved_models"
                     saving_milestone = cfg.train.saving_milestone #5 
                     )
    # start training
    training_losses, validation_losses, lr_rates = trainer_object.run_trainer()  

    data = {'train_losses': training_losses, 
            'val_losses': validation_losses}
    df = pd.DataFrame(data)
    df.to_csv (cfg.train.losses_file, index = False, header=True)

    data = {'lr_rates': lr_rates}
    df = pd.DataFrame(data)
    df.to_csv(cfg.train.lr_rates_file, index = False, header=True) 
    log.info("Training completed")

    fig = inference.plot_training_v(training_losses, validation_losses, lr_rates, gaussian=True, sigma=1, figsize=(10, 4))
    fig.savefig(cfg.train.savefig_png)
    fig.savefig(cfg.train.savefig_pdf)
    plt.close(fig)  
    log.info("Plot saved")  

    model_restored =  models.ResNetReg(
        config = resnet18_4_config, 
        input_dim = cfg.model.in_channels, 
        output_dim = cfg.model.outputs
         ).to(device)

    model_name = cfg.restore.model_name
    model_weights = torch.load(model_name, map_location = device)    
    model_restored.load_state_dict(model_weights)   

    test_batch_size = 2 * cfg.train.batch_size

    transforms_test = Compose([    
        MoveAxis(transform_target = False)
        ])

    dataset_test = dataloader_nmr.RotationDataSet(dtest, df_test, transforms_test, sine = True)
    log.info(f"{len(dataset_test) = }")

    # dataloader test  
    dataloader_test = DataLoader(dataset=dataset_test,
                                     batch_size=test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers, 
                                     pin_memory=True)                                 

    log.info(f"Test loader size: {len(dataloader_test)}")
 

    predictions = inference.predict_batch(
                                        model = model_restored,
                                        test_loader = dataloader_test,
                                        device = device,
                                        notebook=cfg.train.notebook,
                                        dtype = "float",
                                        regression = True 
    )
    predictions = predictions.squeeze() # * np.pi  # casting into radians
    gt = df_test[['sin','cos']].values
    r2 = r2_score(gt, predictions)

    log.info(f"Test R^2 Score: {round(r2, 7)}")  # type: ignore

    pred_angles = np.arctan2(predictions[:, 0], predictions[:, 1])
    gt_angles = df_test['rad'].values
    
    sorted_ind = np.argsort(gt_angles)  # type: ignore
    plt.plot(gt_angles[sorted_ind])
    plt.plot(pred_angles[sorted_ind])
    plt.legend(["gt_angle", "predicted_angle"])    
    plt.title(f"Sorted predictions and gt in angles,  R^2 Score of sine-cosine: {round(r2, 7)}")  # type: ignore
    plt.savefig(cfg.evaluation.savefig_png)
    plt.savefig(cfg.evaluation.savefig_pdf)
    plt.show()

if __name__ == '__main__': 
    main()

    