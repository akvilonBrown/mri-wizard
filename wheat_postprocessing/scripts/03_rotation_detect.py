import logging
import hydra
import inference
from omegaconf import DictConfig, OmegaConf
import dataloader_nmr 

import numpy as np
import os
import pandas as pd
from skimage import io

from collections import namedtuple 
import models
import torch

from torch.utils.data import DataLoader


def load_files(filenames, datafolder, shape = (180, 180)):
    dset = np.zeros(((len(filenames),)+ shape), dtype = np.uint8)
    for i in range(len(filenames)):
        dset[i] = io.imread(os.path.join(datafolder,filenames[i]))
    return np.expand_dims((dset > 0).astype(np.uint8), 1)

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    pca_first_datalist_file = cfg.data.pca_first_datalist_file  
    topfolder_slices = cfg.data.pca_first_debug_folder
    rotation_detected_file_to_save = cfg.data.rotation_detected_file
    rotation_model = cfg.settings.rotation_detection.model_name
    test_batch_size = cfg.settings.rotation_detection.batch_size#16

    num_workers = cfg.settings.rotation_detection.num_workers
    if num_workers == 0 or num_workers > len(os.sched_getaffinity(0)):
       num_workers = len(os.sched_getaffinity(0))
       logger.info(f"num_workers was adjusted to the available cpu count. Count from config file: {cfg.settings.rotation_detection.num_worker}")

    cwd = hydra.utils.get_original_cwd()
    paths =  dataloader_nmr.handle_relative_path(cwd, 
                                        pca_first_datalist_file, 
                                        topfolder_slices, 
                                        rotation_detected_file_to_save,
                                        rotation_model
                                        ) 
    pca_first_datalist_file, topfolder_slices, rotation_detected_file_to_save, rotation_model = paths
    logger.debug(f"{paths = }")    

    structure_df = pd.read_csv(pca_first_datalist_file,  index_col=0)  
    structure_df.reset_index(inplace = True, drop = True)        

    logger.debug("Path to the topfolder: " + topfolder_slices)
    files = list(structure_df.slice_file.values)
     
    loaded_files = load_files(files, topfolder_slices)
    logger.info(f"{loaded_files.shape = }")

    dummy = [0.0]* len(files)
    dummy_data = {
        "filename": files,
        "rad" : dummy,
        "cos" : dummy,
        "sin" : dummy
    }
    dframe = pd.DataFrame(dummy_data)
    dataset_test = dataloader_nmr.RotationDataSet(loaded_files, dframe, sine = True)
    dataloader_test = DataLoader(dataset=dataset_test,
                                         batch_size=test_batch_size,
                                         shuffle=False,
                                         num_workers=num_workers, 
                                         pin_memory=True) 

 

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        logger.error("CUDA unavailable")
        device = torch.device('cpu') 

    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

    resnet18_4_config = ResNetConfig(block = models.BasicBlock,
                                   n_blocks = cfg.settings.rotation_detection.model.n_blocks,
                                   channels = cfg.settings.rotation_detection.model.channels)  

    model_restored = models.ResNetReg(
        config = resnet18_4_config, 
        input_dim = cfg.settings.rotation_detection.model.in_channels, 
        output_dim = cfg.settings.rotation_detection.model.outputs
         ).to(device)                                          

    model_weights = torch.load(rotation_model, map_location = device)    
    model_restored.load_state_dict(model_weights)

    logger.info("Model succesfully loaded")

    predictions = inference.predict_batch(
                                        model = model_restored,
                                        test_loader = dataloader_test,
                                        device = device,
                                        notebook=False,
                                        dtype = "float",
                                        regression = True 
    ) 

    dframe.sin = predictions[:,0]
    dframe.cos = predictions[:,1]
    dframe.rad = np.arctan2(predictions[:, 0], predictions[:, 1])
    dframe["angle"] = np.round(dframe.rad * 180 /np.pi, 0).astype(int)

    df = pd.concat([structure_df, dframe], axis = 1)
    df.drop('filename', axis=1, inplace=True)
    df.to_csv(rotation_detected_file_to_save)
    logger.info("Processing completed")

if __name__ == '__main__':
    main()