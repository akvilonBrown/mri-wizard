import logging
import hydra
from omegaconf import DictConfig, OmegaConf

import dataloader_nmr
import numpy as np
import nibabel as nib
import os
import pandas as pd

from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale

logger = logging.getLogger(__name__)


def load_config_file(path: str) -> pd.DataFrame:
    ending = path.split('.')[-1]
    if "xls" in ending:
        config = pd.read_excel(path,  index_col=0)
    elif ending == "csv":
        config = pd.read_csv(path,  index_col=0)
    else:
        raise Exception("Wrong type of config file")
    config.reset_index(inplace = True)    
    return config

'''
Function to determine the best number of clusters (above 1) 
based on the inertia improvement and threshold relative to the initial inertia
'''
def get_optimal_cluster_points(data, bestnum = None, inertia_threshold = 0.05):
    if bestnum is None:
        inertias = []
        ncluslist = list(range(1,5))
        for i, nclus in enumerate(ncluslist):
            clustering = KMeans(n_clusters=nclus, random_state=0).fit(data)  
            if nclus > ncluslist[0]:
                dd = abs(clustering.inertia_ - inertias[-1])
                relative_diff = dd/inertias[0]                
                if relative_diff < inertia_threshold:
                    bestnum = ncluslist[i-1]
                    break                
            inertias.append(clustering.inertia_)
        if bestnum is None:
            bestnum = ncluslist[-1]    
           
    clustering = KMeans(n_clusters=bestnum, random_state=0).fit(data)
    centers = np.round(clustering.cluster_centers_, 0).astype(int)
    return centers

'''
Function which calculates distance map
nmclosinig - parameter to alter morfology (multiple closing, aimed to remove cavity inside the object)
'''
def get_distmap(lb, nmclosing = 30, inertia_threshold = 0.05, bestnum = None):
    lb_mask = np.where(lb>0, 1, 0)
    if (nmclosing == None):
        lb_maskbin = lb_mask 
    else:   
        lb_maskbin = lb_mask.astype(bool)
        for i in range(nmclosing):
            lb_maskbin = morphology.binary_dilation(lb_maskbin)
            
        for i in range(nmclosing):
            lb_maskbin = morphology.binary_erosion(lb_maskbin)        
        lb_maskbin = lb_maskbin.astype(int)
        
    dist_map = ndi.distance_transform_edt(lb_maskbin)
    logger.debug(f"Shape and max value of distance map: {dist_map.shape, dist_map.max()}")  # type: ignore
    
    lb_e = np.where(lb==1, 1, 0)
    
    lb_ex = np.argwhere(lb_e)
    centers = get_optimal_cluster_points(lb_ex, inertia_threshold = inertia_threshold, bestnum = bestnum)
    nm_inst = centers.shape[0]
    logger.debug(f"Number of clusters: {nm_inst}") 

    mx_peak_x = peak_local_max(dist_map) 
    dist_mx = distance.cdist(mx_peak_x, centers, 'euclidean')
    logger.debug(f"Max distance points shape: {dist_mx.shape}")
    closest_pts = np.zeros((nm_inst, 3), dtype = int)
    
    for i in range(nm_inst):
        closest_pts[i] = mx_peak_x[np.argmin(dist_mx[:, i])]
        
    logger.debug(f"Closest points_pts: {closest_pts}")      
    
    
    voldts_cp = np.zeros_like(lb_e)
    for cp in closest_pts:
        voldts_cp[cp[0], cp[1], cp[2]] = 1  

    mrks, _ = ndi.label(voldts_cp)  # type: ignore
    logger.debug(f"Unique labels {np.unique(mrks)}")  
    
    return dist_map, mrks, lb_mask

def segment_deviate(lbls):
    nm_inst = np.max(np.unique(lbls))    
    lbls_enums=[]
    for i in range(1, nm_inst+1):
        lbls_enums.append(np.count_nonzero(np.where(lbls==i, 1, 0)))
    
    logger.info("Number of voxels per each class below:")
    for i, lbenum in enumerate(lbls_enums):
        logger.info(f"{i+1, lbenum}")
    
    lb_enumsnp = np.array(lbls_enums)
    lb_enumsnp = lb_enumsnp/np.max(lb_enumsnp)
    
    return lb_enumsnp.std()

@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    inst_shape = (cfg.settings.separation.input_dim0, cfg.settings.separation.input_dim1, cfg.settings.separation.input_dim2) 
    im_min =  cfg.settings.separation.im_min 
    prepath = cfg.data.pre 
    save_dir = cfg.data.saved_folder_separated
    dev_threshold = cfg.settings.separation.dev_threshold  #def 0.2
    nmclosing_default = cfg.settings.separation.nmclosing_default
    nmclosing_default = None if nmclosing_default == 0 else nmclosing_default
    nmclosing_alter = cfg.settings.separation.nmclosing_alter
    inertia_threshold = cfg.settings.separation.inertia_threshold
    bestnum = cfg.settings.separation.bestnum
    bestnum = None if bestnum == 0 else bestnum
    rescale_required = cfg.settings.separation.rescale.required
    rescale_range = cfg.settings.separation.rescale.range
    config_file_separated = cfg.data.config_file_separated
    config_file = cfg.data.config_file
    strip_pericarp = cfg.data.strip_pericarp
    
     # handling relative path
    cwd = hydra.utils.get_original_cwd()   
    paths =  dataloader_nmr.handle_relative_path(cwd, 
                                        config_file, 
                                        prepath, 
                                        save_dir,
                                        config_file_separated
                                        ) 
    config_file, prepath, save_dir, config_file_separated = paths
    logger.info(f"{paths = }")

    
    datafr = load_config_file(config_file)
    
    cultivar_col = []
    folder_col = []
    source_col = []
    label_col = []

    for di in range(len(datafr)): 
        folder =  str(datafr.loc[di, "folder"])     
        logger.info("Processing item " + folder) 
        #if(di>3):
        #    break
        savepath = os.path.join(save_dir, folder)
        if not os.path.exists(savepath):
            os.makedirs(savepath) 
            logger.info(f"Target folder created {savepath}")          

    
        init_path = os.path.join(prepath, folder)
        try:        
            source_p = os.path.join(init_path, str(datafr.loc[di, "source_file"]))                
            label_p = os.path.join(init_path,  str(datafr.loc[di, "label_file"]))          
            logger.info(f"Source: {str(datafr.loc[di, 'source_file'])}  label: {str(datafr.loc[di, 'label_file'])}") 
            
            lb_nii = nib.load(label_p)
            lb_np = lb_nii.get_fdata(dtype=np.float32).astype(np.uint8)
            if strip_pericarp:
                lb_np = np.where(lb_np<4, lb_np, 0)
            
            distance_map, markers, lb_npbin_init = get_distmap(lb_np, nmclosing = nmclosing_default, inertia_threshold = inertia_threshold, bestnum = bestnum)  # type: ignore
            labels = watershed(-distance_map, markers, mask=lb_npbin_init)  # type: ignore
            num_labels = np.max(labels)                
            logger.info(f"Labels shape {labels.shape} and max number {num_labels}")
            dev = segment_deviate(labels)
            logger.debug(f"STD of separated volumes: {dev}")                
            
            # deviation threshold to determine if volume distribution between instance is similar            
            if dev>dev_threshold:
                logger.warning("Uneven volume distribution, trying alternative approach with morphology")
                distance_map, markers, lb_npbin_init = get_distmap(lb_np, nmclosing = nmclosing_alter, inertia_threshold = inertia_threshold, bestnum = bestnum)
                labels = watershed(-distance_map, markers, mask=lb_npbin_init)  # type: ignore
                num_labels = np.max(labels)                
                logger.info(f"Labels shape {labels.shape} and max number {num_labels}")
                dev = segment_deviate(labels)
                logger.debug(f"STD of separated volumes: {dev}") 
                if dev>dev_threshold:  # just one alternative attempt
                    logger.error("Uneven volume distribution, instance segmentation unsuccesfull. Skipping this item, please retry it with different morphology settings")
                    logger.error(f"item: {folder}  label: { str(datafr.loc[di, 'label'])}")
                    continue                    
            
            logger.info("Instance segmentation successful") 
            img_nii = nib.load(source_p)
            img_np = img_nii.get_fdata(dtype=np.float32)
            img_np = np.squeeze(img_np)
            if rescale_required:
                shape = img_np.shape
                img_np = minmax_scale(img_np.ravel(), feature_range=rescale_range).reshape(shape)
            
            obfound = ndi.find_objects(labels)
            for i in range(num_labels):
                file_pre = str(datafr.loc[di, "source_file"]).split(".nii", 1)[0]
                file_end = str(datafr.loc[di, "source_file"]).split(".", 1)[-1] # must be .nii, but also .gz.
                label_pre = str(datafr.loc[di, "label_file"]).split(".nii", 1)[0]
                label_end = str(datafr.loc[di, "label_file"]).split(".", 1)[-1] # must be .nii, but also .gz.                
                
                file_end = ".nii.gz" if "gz" in file_end else ".nii"
                label_end = ".nii.gz" if "gz" in label_end else ".nii"
            
                logger.debug(f"Processing instance {i}")
                obfound_loc = obfound[i]
                # creating a volumetric mask from a single instance
                label_mask = np.where(labels==i+1, 1,0)
            
                # shrinking the volume borders to the actual object to determine the placement further in a new volume
                ob = label_mask[obfound_loc] 
            
                #assert (inst_shape[0]>=ob.shape[0] and inst_shape[1]>=ob.shape[1] and inst_shape[2]>=ob.shape[2]), "Object dimension is bigger" + str(ob.shape)
                if (inst_shape[0]<ob.shape[0] or inst_shape[1]<ob.shape[1] or inst_shape[2]<ob.shape[2]):
                    logger.error(f"Object dimention {ob.shape} appeared bigger than allocated volume {inst_shape}. Skipping this label")
                    continue
                
                x_center = ob.shape[0]//2
                xpad1 = inst_shape[0]//2-x_center
                xpad2 = inst_shape[0]-(xpad1+ob.shape[0])
                
                y_center = ob.shape[1]//2
                ypad1 = inst_shape[1]//2-y_center
                ypad2 = inst_shape[1]-(ypad1+ob.shape[1])
                
                z_center = ob.shape[2]//2
                zpad1 = inst_shape[2]//2-z_center
                zpad2 = inst_shape[2]-(zpad1+ob.shape[2])    
                logger.debug(f"Pads {xpad1, xpad2, ypad1, ypad2, zpad1, zpad2}")    
            
                # carving the labels for a single instance from the big volume
                lb_inst = np.where(label_mask, lb_np, 0)
                ob_inst = lb_inst[obfound_loc]
                ob_cube = np.pad(ob_inst, ((xpad1, xpad2), (ypad1, ypad2), (zpad1, zpad2)), 'constant', constant_values=0)
                ob_cube = ob_cube.astype(dtype = 'uint8')
                ob_cube_nifty = nib.Nifti1Image(ob_cube, affine=np.eye(4))
                ob_cube_nifty.header['pixdim'] = lb_nii.header['pixdim']  # type: ignore
                filesave_label = label_pre + "_label_instance_" + str(i+1) + label_end
                nib.save(ob_cube_nifty, os.path.join(savepath, filesave_label))
                
                # carving the source signal for a single instance from the big volume
                img_masked = np.where(label_mask, img_np, im_min)
                img_masked_obj = img_masked[obfound_loc]
                img_cube = np.pad(img_masked_obj, ((xpad1, xpad2), (ypad1, ypad2), (zpad1, zpad2)), 'constant', constant_values=im_min)
                img_cube =  np.expand_dims(img_cube, 3)
                img_cube = img_cube.astype(np.int16)
                img_cube_nifty = nib.Nifti1Image(img_cube, affine=np.eye(4))
                img_cube_nifty.header['pixdim'] = img_nii.header['pixdim']  # type: ignore
                filesave_source = file_pre + "_source_instance_" + str(i+1) + file_end
                nib.save(img_cube_nifty, os.path.join(savepath, filesave_source))

                cultivar_col.append(datafr.loc[di, "cultivar"])
                folder_col.append(folder) 
                source_col.append(filesave_source) 
                label_col.append(filesave_label)
                
        except Exception as e:
            logger.error(f"Error occured with this sample: {folder}") 
            logger.exception(e)  
    
    data = {
        "cultivar" : cultivar_col,
        "folder" : folder_col,
        "source_instance" : source_col,
        "label_instance" : label_col                
    }

    df = pd.DataFrame(data)
    df.to_csv(config_file_separated, header=True)

if __name__ == '__main__':
    main()