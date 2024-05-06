import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import dataloader_nmr
import numpy as np
import os
import pandas as pd
from skimage import io, measure, morphology


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg : DictConfig) -> None:

    crease_depth_topfolder = cfg.data.crease_depth_topfolder
    if len(crease_depth_topfolder)  == 0:
        crease_depth_topfolder = cfg.data.pca_final_debug_folder

    crease_depth_config_file =  cfg.data.crease_depth_config_file
    if len(crease_depth_config_file) == 0:
        crease_depth_config_file = cfg.data.pca_final_datalist_file

    crease_depth_targetfolder = cfg.data.crease_depth_targetfolder
    crease_depth_result_file = cfg.data.crease_depth_result_file



    draw_value = cfg.settings.crease_depth.draw_value
    morphology_value = cfg.settings.crease_depth.morphology_value
    morphology_value_reduction = cfg.settings.crease_depth.morphology_value_reduction
    
    cwd = hydra.utils.get_original_cwd()
    paths = dataloader_nmr.handle_relative_path(cwd, 
                                        crease_depth_topfolder, 
                                        crease_depth_config_file, 
                                        crease_depth_targetfolder,
                                        crease_depth_result_file
                                        ) 
    crease_depth_topfolder, crease_depth_config_file, crease_depth_targetfolder, crease_depth_result_file = paths 
    if not os.path.exists(crease_depth_targetfolder):
        os.makedirs(crease_depth_targetfolder, exist_ok=True)
        logger.info(f"Target folder {crease_depth_targetfolder} created") 

    structure_df = pd.read_csv(crease_depth_config_file,  index_col=0)  
    structure_df.reset_index(inplace = True, drop = True)
    structure_df["crease_depth"] = ""    

    logger.debug("Path to the topfolder: " + crease_depth_topfolder)

    props = [
        'image_convex',
        'image_filled',
        'bbox'    
        ]

    for di in range(len(structure_df)):
        try: 
            file = str(structure_df.loc[di, "slice_file"])
            logger.info(f"Processing item {file}")

            file_path_source = os.path.join(crease_depth_topfolder, file)
            file_path_target = os.path.join(crease_depth_targetfolder, file)            

            img_init = io.imread(file_path_source)
            img = (img_init > 0)
            img = img.astype(np.uint8)
            retrived_props = measure.regionprops_table(
                    img,
                    intensity_image=None,
                    properties=props,
                )
            imgconv = retrived_props['image_convex'][0] #.astype(np.uint8)
            imgfill = retrived_props['image_filled'][0] #.astype(np.uint8)
            bbox0, bbox1, bbox2, bbox3 = retrived_props['bbox-0'][0], retrived_props['bbox-1'][0], retrived_props['bbox-2'][0], retrived_props['bbox-3'][0]
            diff = imgconv ^ imgfill
            morpho = morphology.remove_small_objects(diff, morphology_value)
            if np.count_nonzero(morpho) == 0:
                morphology_value_reduced = morphology_value
                while (np.count_nonzero(morpho) == 0 and morphology_value_reduced > 0) :
                    logger.info(f"----Reducing morphology requirements to the item {file}")
                    morphology_value_reduced -= morphology_value_reduction
                    morpho = morphology.remove_small_objects(diff, morphology_value_reduced)        

            ixx = np.argwhere(morpho)
            cd = np.max(ixx[:,0]) - np.min(ixx[:,0])
            structure_df.loc[di, "crease_depth"] = cd
            vmin = np.argmin(ixx[:,0])
            xx = ixx[vmin,1]
            yy = np.min(ixx[:,0])
            imbb_init = img_init[bbox0:bbox2, bbox1:bbox3]
            imbb_init[yy] = draw_value    
            imbb_init[yy:, xx] = draw_value
            io.imsave(file_path_target, imbb_init)

        except Exception as e:
            logger.error(f"Error occured with row: {di}")
            logger.exception(e)    

    structure_df.to_csv(crease_depth_result_file)

if __name__ == '__main__':
    main()