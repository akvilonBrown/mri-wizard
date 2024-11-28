import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import dataloader_nmr
import numpy as np
import nibabel as nib
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import io

from monai.transforms.spatial.array import Affine
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


def load_nifty(source_pt, dtype = np.uint8, pad = None):    
    img_nii = nib.load(source_pt)
    img_npt = img_nii.get_fdata(dtype=np.float32)
    img_npt = np.squeeze(img_npt)
    if pad is not None:
        img_npt = np.pad(img_npt, ((pad, pad), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    img_npt = img_npt.astype(dtype)
    return img_npt #np.where(img_npt>0, 1, 0)

def show_central_slice(array, show = True):
    cen = array.shape[0]//2
    picx = array[cen]    
    if show:
        plt.imshow(picx)
        plt.show()
    return  picx 

def show_longitudal_slice(array, show = True):
    cen = array.shape[1]//2
    picx = array[:, cen]    
    if show:
        plt.imshow(picx)
        plt.show()
    return  picx
    
def calculate_pca(array):
    array = np.where(array>0, 1, 0)
    mono = np.argwhere(array)
    pca = PCA(n_components=3) #choose the number of components = number of original dimensions
    monoz_new = pca.fit_transform(mono)
    pca_c = pca.components_  # type: ignore
    pca_v = pca.explained_variance_  # type: ignore
    return pca_c, pca_v, monoz_new
    
def rotate(array, matrix, mode = "nearest", extra = None):
    assert mode in ["bilinear","nearest" ], "mode should be bilinear or nearest"
    if extra is not None:
        matrix = extra @ matrix
    newm = np.zeros((4,4))
    newm[:3, :3] = matrix.T
    newm[3, 3] = 1
    affine = Affine(
        affine = newm,
        mode=mode, 
        padding_mode="zeros")
    arrayd = np.expand_dims(array, 0)
    arrayd_rotated = affine(arrayd)[0]  # type: ignore
    return arrayd_rotated.numpy().squeeze()

def is_embryo_location_right(array):
    embr = np.where(array.astype(np.uint8) == 1, 1,0)
    argw = np.argwhere(embr)
    return argw[:, 0].mean() > embr.shape[0]//2

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg : DictConfig) -> None:
    
    if  len(cfg.data.config_file_separated_pca_final) == 0:
        config_file_separated = cfg.data.rotation_detected_file        
    else: 
        config_file_separated = cfg.data.config_file_separated_pca_final 

    pca_final_datalist_file = cfg.data.pca_final_datalist_file 
    topfolder_source = cfg.data.saved_folder_separated
    topfolder_target = cfg.data.pca_final_target
    debug = cfg.settings.pca_final.debug  
    topfolder_target_debug = cfg.data.pca_final_debug_folder
    dochecks = cfg.settings.pca_final.dochecks 
    pad = cfg.settings.pca_final.extra_pad
    if pad <= 0:
        pad = None     

    
    cwd = hydra.utils.get_original_cwd()
    paths = dataloader_nmr.handle_relative_path(cwd, 
                                        config_file_separated, 
                                        pca_final_datalist_file, 
                                        topfolder_source,
                                        topfolder_target,                                        
                                        topfolder_target_debug
                                        ) 
    config_file_separated, pca_final_datalist_file, topfolder_source, topfolder_target, topfolder_target_debug = paths 
    logger.debug(f"{paths = }")
    structure_df = pd.read_csv(config_file_separated,  index_col=0)  
    structure_df.reset_index(inplace = True, drop = True)
    structure_df["slice_file"] = ""


    rot90 = np.array([[ 1.,  0.,  0.],
                    [ 0.,  0., -1.],
                    [ 0.,  1.,  0.]])

    for di in range(len(structure_df)):
        try: 
            folder = str(structure_df.loc[di, "folder"])
           
            folder_target = os.path.join(topfolder_target,folder)
            if not os.path.exists(folder_target):
                os.makedirs(folder_target, exist_ok=True)
                logger.info(f"Target folder {folder} created")  

            lfile = str(structure_df.loc[di, "label_instance"])
            sfile = str(structure_df.loc[di, "source_instance"])
            
            logger.info(f"Processing item {folder} {lfile}")

            sfile_path_source = os.path.join(topfolder_source, folder, sfile)
            lfile_path_source = os.path.join(topfolder_source, folder, lfile)
            sfile_path_target = os.path.join(topfolder_target, folder, sfile)
            lfile_path_target = os.path.join(topfolder_target, folder, lfile)              

            label = load_nifty(lfile_path_source,  dtype = np.uint8, pad = pad)
            source = load_nifty(sfile_path_source, dtype = np.int16, pad = pad) 
            matr, _, _ = calculate_pca(label)
            angle_rad = (structure_df.loc[di, "rad"])
            r = R.from_euler('xyz', (-angle_rad, 0, 0), degrees=False) # type: ignore
            rmat_roll = r.as_matrix()
            matr = rmat_roll @ rot90 @ matr

            label_aligned = rotate(label, matr, mode = "nearest").astype(np.uint8) #no extra rotation 90 - it's included into the main matrix
            source_aligned = rotate(source, matr, mode = "bilinear").astype(np.int16)
            if dochecks:
                # checking if reflection has taken place
                det = round(np.linalg.det(matr))
                if det < 0:
                    print("----------flipped")
                    label_aligned = np.flip(label_aligned, 2)
                    source_aligned = np.flip(source_aligned, 2)

                # ensuring that embryo is located on the same side of the image
                if is_embryo_location_right(label_aligned):
                    print("----------flipped to correct embryo position")
                    label_aligned = np.flip(label_aligned, 0)
                    label_aligned = np.flip(label_aligned, 2) 
                    source_aligned = np.flip(source_aligned, 0)
                    source_aligned = np.flip(source_aligned, 2)

            
            label_aligned_nifty = nib.Nifti1Image(label_aligned, affine=np.eye(4))
            source_aligned_nifty = nib.Nifti1Image(source_aligned, affine=np.eye(4))
            nib.save(source_aligned_nifty, sfile_path_target) 
            nib.save(label_aligned_nifty, lfile_path_target)

            if debug:  #show_central_slice(mololith_aligned)
                if not os.path.exists(topfolder_target_debug):
                    os.makedirs(topfolder_target_debug, exist_ok=True)
                    logger.info(f"Target folder {topfolder_target_debug} created")                
                imfile = folder + "_" + lfile.split('.nii')[0] + ".png"                
                imfile_longitudal = folder + "_" + lfile.split('.nii')[0] + "_longitudal.png" 
                imfile_path = os.path.join(topfolder_target_debug, imfile) 
                imfile_path_longitudal = os.path.join(topfolder_target_debug, imfile_longitudal) 
                slice_ = show_central_slice(label_aligned, show = False)
                slice_ = (np.round(slice_/4 * 255)).astype(np.uint8)
                io.imsave(imfile_path, slice_)
                slice_ = show_longitudal_slice(label_aligned, show = False)
                slice_ = (np.round(slice_/4 * 255)).astype(np.uint8)
                io.imsave(imfile_path_longitudal, slice_)
                structure_df.loc[di, "slice_file"] = imfile

        except Exception as e:
            logger.error(f"Error occured with row: {di}")
            logger.exception(e)   

    structure_df.to_csv(pca_final_datalist_file)

if __name__ == '__main__':
    main()