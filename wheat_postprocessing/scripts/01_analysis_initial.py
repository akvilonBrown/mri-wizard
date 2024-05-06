'''
Initial analysis of segmented and separated samples
Sometimes this script may require a high amount of RAM
'''

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

def analize_sample(pre, source_name, label_name, mono_props, rest_props, cols, morphology_pix = 50, with_pericarp = True):
    source_p = os.path.join(pre, source_name)
    label_p = os.path.join(pre, label_name)
    
    img_nii = nib.load(source_p)
    img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))
    
    lb_nii = nib.load(label_p)
    lb_np = lb_nii.get_fdata(dtype=np.float32).astype(np.uint8)    

    if not with_pericarp:
        lb_np = np.where(lb_np<4, lb_np, 0)
    lb_monolith = lb_np>0

    lb_monolith = morphology.remove_small_objects(lb_monolith, 50)
    lb_monolith = morphology.remove_small_holes(lb_monolith, 50)
    lb_monolith = lb_monolith.astype(np.uint8)
    
    info_table_monolith = pd.DataFrame(
        measure.regionprops_table(
            lb_monolith,
            intensity_image=img_np,
            properties=mono_props,
        )
    ).set_index('label')

    verts, faces, normals, values = measure.marching_cubes(lb_monolith) # type: ignore    
    surf_area = measure.mesh_surface_area(verts, faces)
        
    info_table_monolith["surface_area"] = [surf_area]
    totvol = info_table_monolith.loc[1, 'area']
    info_table_monolith["volume_ratio"] = info_table_monolith['area']/totvol  # type: ignore       

    
    lb_emb = lb_np==1
    lb_emb = morphology.remove_small_objects(lb_emb, morphology_pix)
    lb_emb = morphology.remove_small_holes(lb_emb, morphology_pix)
    lb_emb = lb_emb.astype(np.uint8)
    
    info_table_e = pd.DataFrame(
        measure.regionprops_table(
            lb_emb,
            intensity_image=img_np,
            properties=mono_props,
        )
    ).set_index('label')
    
    verts, faces, normals, values = measure.marching_cubes(lb_emb) # type: ignore    
    surf_area = measure.mesh_surface_area(verts, faces)
    
    info_table_e["surface_area"] = [surf_area]
    info_table_e["volume_ratio"] = info_table_e['area']/totvol # type: ignore    
    
        
    lb_rest= np.where(lb_np>1, lb_np,0)    
    info_table_rest = pd.DataFrame(
        measure.regionprops_table(
            lb_rest,
            intensity_image=img_np,
            properties=rest_props,
        )
    ).set_index('label')
    
    info_table_rest["volume_ratio"] = info_table_rest['area']/totvol # type: ignore    
    
   
    info_table_monolith['label'] = ["monolith"]
    info_table_e['label'] = ["embryo"]

    if with_pericarp:
        info_table_rest['label'] = ["endosperm", "aleuron", "pericarp"]
    else:
        info_table_rest['label'] = ["endosperm", "aleuron"]     
     
     
    data = [info_table_monolith, info_table_e, info_table_rest]
    vertical_concat = pd.concat(data)  #ignore_index = True
    vertical_concat.set_index('label', inplace = True)
    vertical_concat["class"] = vertical_concat.index
    
    summary = vertical_concat[cols]
    return summary  

mono_props = [
'area',
'intensity_max',
'intensity_mean',
'intensity_min',    
'area_convex',
'area_filled',
'axis_major_length',
'axis_minor_length',
'equivalent_diameter_area',
'euler_number',
'feret_diameter_max',
'label',
'solidity',
'extent'    
]

rest_props = [
'area',
'intensity_max',
'intensity_mean',
'intensity_min',
'label'
]

cols = [
#'label', 
#'filename',
'class',    
'area',
'surface_area',
'volume_ratio',
'intensity_max',
'intensity_mean',
'intensity_min',    
'area_convex',
'area_filled',
'axis_major_length',
'axis_minor_length',
'equivalent_diameter_area',
'euler_number',
'feret_diameter_max',
'solidity',
'extent'    
]


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg : DictConfig) -> None:
    topfolder = cfg.data.saved_folder_separated  
    config_file_separated = cfg.data.config_file_separated
    analysis_initial_file = cfg.data.analysis_initial_file
    save_folder_results = cfg.data.save_folder_results
    save_folder_results_file = cfg.data.save_folder_results_file
    include_pericarp = cfg.data.include_pericarp 

  
    cwd = hydra.utils.get_original_cwd()    
    # handling relative path
    topfolder, config_file_separated, analysis_initial_file = dataloader_nmr.handle_relative_path(cwd, 
                                        topfolder, 
                                        config_file_separated, 
                                        analysis_initial_file                                        
                                        )
    logger.info("Path to the topfolder: " + topfolder)
    structure_df = pd.read_csv(config_file_separated,  index_col=0)  
    structure_df.reset_index(inplace = True, drop = True)

    dframes = []

    for di in range(len(structure_df)):   
        try: 
            folder = str(structure_df.loc[di, "folder"])
           
            pre = os.path.join(topfolder,folder)
            lfile = str(structure_df.loc[di, "label_instance"])
            sfile = str(structure_df.loc[di, "source_instance"])
            logger.info(f"Processing item {folder} {lfile} {sfile}")

            dframe = analize_sample(pre = pre, source_name=sfile, label_name=lfile, 
                            mono_props=mono_props, rest_props=rest_props, 
                            cols = cols, with_pericarp = include_pericarp)
            
            sorted_cols = ["cultivar", "folder", "source_instance", "label_instance"] + list(dframe.columns)
            dframe["cultivar"] = structure_df.loc[di, "cultivar"]
            dframe["folder"] = folder
            dframe["source_instance"] = sfile
            dframe["label_instance"] = lfile
            dframe.reset_index(inplace = True)            
            dframe = dframe[sorted_cols]
            dframes.append(dframe)
            
            if save_folder_results:  
                subsummary = pd.concat(dframes, ignore_index = True)
                df_slice = subsummary[subsummary.folder==folder]                 
                df_slice.to_csv(os.path.join(pre, save_folder_results_file))


        except Exception as e:
            logger.error(f"Error occured with row: {di}")
            logger.exception(e)   

    summary = pd.concat(dframes, ignore_index = True)
    summary.to_csv(analysis_initial_file)

if __name__ == '__main__':
    main()