import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import dataloader_nmr
import numpy as np
import nibabel as nib
import os

import pandas as pd
from skimage import measure, morphology

logger = logging.getLogger(__name__)


def get_bbox(pre, label_name, morphology_pix = 50):   
    
    label_p = os.path.join(pre, label_name)
    lb_nii = nib.load(label_p)
    lb_np = lb_nii.get_fdata(dtype=np.float32).astype(np.uint8)    

    lb_monolith = lb_np>0
    lb_monolith = morphology.remove_small_objects(lb_monolith, morphology_pix)
    lb_monolith = morphology.remove_small_holes(lb_monolith, morphology_pix)
    lb_monolith = lb_monolith.astype(np.uint8)
    
    rp = measure.regionprops_table(
            lb_monolith,            
            properties=["bbox"],
         )
    return rp['bbox-3'][0]-rp['bbox-0'][0], rp['bbox-4'][0]-rp['bbox-1'][0], rp['bbox-5'][0]-rp['bbox-2'][0] 


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg : DictConfig) -> None:
    analysis_bbox_topfolder = cfg.data.analysis_bbox_topfolder  
    if len(analysis_bbox_topfolder) == 0:
        analysis_bbox_topfolder = cfg.data.pca_final_target 
 
    analysis_bbox_config_file  = cfg.data.analysis_bbox_config_file  
    if len(analysis_bbox_config_file) == 0:
        analysis_bbox_config_file = cfg.data.crease_depth_result_file 

    analysis_bbox_result_file = cfg.data.analysis_bbox_result_file

    cwd = hydra.utils.get_original_cwd() 
    paths = dataloader_nmr.handle_relative_path(cwd, 
                                        analysis_bbox_topfolder, 
                                        analysis_bbox_config_file, 
                                        analysis_bbox_result_file
                                        ) 
    analysis_bbox_topfolder, analysis_bbox_config_file, analysis_bbox_result_file = paths
    logger.debug(f"{paths = }")

    structure_df = pd.read_csv(analysis_bbox_config_file,  index_col=0)  
    structure_df.reset_index(inplace = True, drop = True)
    structure_df["length"] = ""
    structure_df["width"] = ""
    structure_df["depth"] = ""  

    for di in range(len(structure_df)): 
        try: 
            folder = str(structure_df.loc[di, "folder"])
           
            pre = os.path.join(analysis_bbox_topfolder,folder)
            lfile = str(structure_df.loc[di, "label_instance"])
            logger.info(f"Processing item {folder} {lfile}")

            l, d, w = get_bbox(pre = pre, label_name=lfile)
            structure_df.loc[di, "length"] = l
            structure_df.loc[di, "width"] = w
            structure_df.loc[di, "depth"] = d

        except Exception as e:
            logger.error(f"Error occured with row: {di}")
            logger.exception(e)  

    structure_df.to_csv(analysis_bbox_result_file)

if __name__ == '__main__':
    main()