import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import dataloader_nmr
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg : DictConfig) -> None:
    analysis_initial_file = cfg.data.analysis_initial_file
    analysis_bbox_result_file = cfg.data.analysis_bbox_result_file
    result_file = cfg.data.result_file
    pixdim = cfg.data.resolution

    cwd = hydra.utils.get_original_cwd() 
    paths = dataloader_nmr.handle_relative_path(cwd, 
                                        analysis_initial_file, 
                                        analysis_bbox_result_file, 
                                        result_file
                                        ) 
    analysis_initial_file, analysis_bbox_result_file, result_file = paths
    logger.debug(f"{paths = }")



    base_df = pd.read_csv(analysis_initial_file, index_col=[0])
    df_aligned = pd.read_csv(analysis_bbox_result_file, index_col=[0])

    df_aligned["join"] = df_aligned.folder + "_" + df_aligned.label_instance
    base_df["join"] = base_df.folder + "_" + base_df.label_instance

    merged_df =  base_df.merge(df_aligned, how='left', 
                                             left_on='join', 
                                             right_on='join',
                                             suffixes=('','_aligned'))

    merged_df.drop(['join', 'cultivar_aligned', 'folder_aligned', 'source_instance_aligned',
           'label_instance_aligned', 'slice_file', 'rad', 'cos', 'sin', 'angle' ], axis=1, inplace=True)
    merged_df.loc[merged_df['class'] != 'monolith',  ['length', 'width', 'depth', 'crease_depth']] = None

    #Cast to IU
    df_copy = merged_df.copy()
    df_original = merged_df

    df_copy.area = np.round(df_original.area * pow(pixdim,3), 3)
    df_copy.surface_area =  np.round(df_original.surface_area * pow(pixdim,2), 3)
    df_copy.area_convex =  np.round(df_original.area_convex * pow(pixdim,3), 3)
    df_copy.area_filled =  np.round(df_original.area_filled * pow(pixdim,3), 3)
    df_copy.area_convex =  np.round(df_original.area_convex * pow(pixdim,3), 3)

    df_copy.axis_major_length =  np.round(df_original.axis_major_length * pixdim, 3)
    df_copy.axis_minor_length =  np.round(df_original.axis_minor_length * pixdim, 3)

    df_copy.equivalent_diameter_area =  np.round(df_original.equivalent_diameter_area * pixdim, 3)
    df_copy.feret_diameter_max =  np.round(df_original.feret_diameter_max * pixdim, 3)

    df_copy.length =  np.round(df_original.length * pixdim, 3)
    df_copy.width =  np.round(df_original.width * pixdim, 3)
    df_copy.depth =  np.round(df_original.depth * pixdim, 3)
    df_copy.crease_depth =  np.round(df_original.crease_depth * pixdim, 3)
    df_copy['sphericity'] =  np.round((np.pi**(1/3)*(6*df_copy.area)**(2/3))/ df_copy.surface_area, 3)

    df_copy.to_excel(result_file)
    logger.info("Completed")


if __name__ == '__main__':
    main()