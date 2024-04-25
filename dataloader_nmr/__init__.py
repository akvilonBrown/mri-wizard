from .twodimdatadet import *
from .transformations import *
from .tripletdataset import *
from .multidataset import *
from .rotationdataset import *

import nibabel as nib
import numpy as np
import os

def normalize_intensity(img_tensor, normalization="mean", percentile=0.8):
    """
    Accepts an ndarray and normalizes it
    :param normalization: choices = "max", "mean" "log10", "clip", type=str
    For mean normalization we use the non zero voxels only.
    source https://theaisummer.com/medical-image-processing/

    :param img_tensor: ndarray with 3D sample, floating point
    :param normalization: normalization type, string
    :param percentile:  percentile to clip the upper bound of intensities 
    return normilized numpy array of input shape
    """
    assert normalization in ["max", "mean", "log10", "clip", "none"], "Normalization should be one of max, mean log10, clip"
    if normalization == "mean":
        mask = np.not_equal(img_tensor, 0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        MAX, MIN = img_tensor.max(), img_tensor.min()
        img_tensor = (img_tensor - MIN) / (MAX - MIN)
    elif normalization == "log10":
        img_tensor =np.clip(img_tensor, a_min = 1e-4, a_max = img_tensor.max())
        img_tensor = np.log10(img_tensor)
    elif normalization == "clip":
        MAX, MIN = img_tensor.max(), img_tensor.min()    
        th = percentile*MAX
        img_tensor =np.clip(img_tensor, a_min = 0, a_max = th)
        MAX, MIN = img_tensor.max(), img_tensor.min()
        img_tensor = (img_tensor - MIN) / (MAX - MIN)        
        
    return img_tensor


'''
Preparing 2D dataset from config file and 3D samples, normalizing (standartizing) within each 3d sample
Working with Nifty data format
    :param prepath: starting path to the folder with data, in this folders samples are placed in subfolders 
according to datastr configuration
    :param datastr: json-like configuration (lists and dictionaries) describing input data structure
    :param percentile:  percentile to clip the upper bound of intensities if normalisation method is clip
    return 3 pairs of data 3D arrays: train set, validation set, and test set

'''
def get_pilot_set(prepath, datastr, normalization = "mean", percentile=0.8):
    train_list_source = []
    val_list_source = []
    test_list_source = []
    train_list_target = []
    val_list_target = []
    test_list_target = []    
    for ln in datastr.keys():        
        vols = datastr[ln] #
        init_path = os.path.join(prepath, ln)
        for vol in vols:
            init_path = os.path.join(prepath, ln)
            source_p = os.path.join(init_path,  vol["source"])
            img_nii = nib.load(source_p)
            #img_nii = nib.as_closest_canonical(img_nii)
            img_np = img_nii.get_fdata(dtype=np.float32)
            if vol["transpose"]:
                img_np = np.moveaxis(img_np, 0, 1)
            rn_from, rn_to = vol["range"][0], vol["range"][1]+1				
            img_np = img_np[rn_from:rn_to]
            img_np_norm = normalize_intensity(img_np, normalization, percentile)
            if(len(img_np_norm.shape)==3): #source shape shoud be [x, 180, 180, 1]
                img_np_norm = np.expand_dims(img_np_norm, -1)
            assert len(img_np_norm.shape) == 4 , "Incorrect source array shape"            
            label_p = os.path.join(init_path,  vol["label"])
            lb_nii = nib.load(label_p)
            lb_np = lb_nii.get_fdata(dtype=np.float32).astype(np.uint8)
            if vol["transpose"]:
                 lb_np = np.moveaxis(lb_np, 0, 1)			
            lb_np = lb_np[rn_from:rn_to]
            assert np.array_equal(np.unique(lb_np), np.array([0, 1, 2, 3, 4])), "Incorrect label values, expected [0, 1, 2, 3, 4] " + label_p
            if(len(lb_np.shape)==4): #target shape shoud be [x, 180, 180]
                lb_np = np.squeeze(lb_np)
            assert len(lb_np.shape) == 3 , "Incorrect target array shape" 
            if(vol["split"] == "test"):
                test_list_source.append(img_np_norm)
                test_list_target.append(lb_np)
            elif (vol["split"] == "validation"):  
                val_list_source.append(img_np_norm)
                val_list_target.append(lb_np)
            else:  
                train_list_source.append(img_np_norm)
                train_list_target.append(lb_np)
                
  
    source_train = np.concatenate(train_list_source)  
    source_val = np.concatenate(val_list_source)   
    source_test = np.concatenate(test_list_source) 
    target_train = np.concatenate(train_list_target)  
    target_val = np.concatenate(val_list_target)   
    target_test = np.concatenate(test_list_target)     
    
    return source_train, target_train, source_val, target_val, source_test, target_test    

def get_dataset(prepath, datastr, split, normalization = "mean", percentile=0.8):
    assert split == "train" or split == "validation" or split == "test", "Only train/validation/test splits are allowed"
    list_source = []
    list_target = []
   
    for ln in datastr.keys():        
        vols = datastr[ln] #
        init_path = os.path.join(prepath, ln)
        for vol in vols:
            if(vol["split"] == split):
                init_path = os.path.join(prepath, ln)
                source_p = os.path.join(init_path,  vol["source"])
                img_nii = nib.load(source_p)
                #img_nii = nib.as_closest_canonical(img_nii)
                img_np = img_nii.get_fdata(dtype=np.float32)
                if vol["transpose"]:
                    img_np = np.moveaxis(img_np, 0, 1)
                rn_from, rn_to = vol["range"][0], vol["range"][1]+1                
                img_np = img_np[rn_from:rn_to]
                img_np_norm = normalize_intensity(img_np, normalization, percentile)
                if(len(img_np_norm.shape)==3): #source shape shoud be [x, 180, 180, 1]
                    img_np_norm = np.expand_dims(img_np_norm, -1)
                assert len(img_np_norm.shape) == 4 , "Incorrect source array shape"            
                label_p = os.path.join(init_path,  vol["label"])
                lb_nii = nib.load(label_p)
                lb_np = lb_nii.get_fdata(dtype=np.float32).astype(np.uint8)
                if vol["transpose"]:
                     lb_np = np.moveaxis(lb_np, 0, 1)            
                lb_np = lb_np[rn_from:rn_to]
                assert np.array_equal(np.unique(lb_np), np.array([0, 1, 2, 3, 4])), "Incorrect label values, expected [0, 1, 2, 3, 4] " + label_p
                if(len(lb_np.shape)==4): #target shape shoud be [x, 180, 180]
                    lb_np = np.squeeze(lb_np)
                assert len(lb_np.shape) == 3 , "Incorrect target array shape"            
                list_source.append(img_np_norm)
                list_target.append(lb_np)                
  
    source = np.concatenate(list_source)
    target = np.concatenate(list_target)   
    
    return source, target

def get_dataset_ext(prepath, datastr, split, normalization = "mean", percentile=0.8):
    assert split == "train" or split == "validation" or split == "test", "Only train/validation/test splits are allowed"
    list_source = []
    list_target = []
   
    for ln in datastr.keys():        
        vols = datastr[ln] #
        init_path = os.path.join(prepath, ln)
        for vol in vols:            
            if (vol["use"]):
                if(vol["split"] == split):
                    init_path = os.path.join(prepath, ln)
                    source_p = os.path.join(init_path,  vol["source"])
                    img_nii = nib.load(source_p)
                    #img_nii = nib.as_closest_canonical(img_nii)
                    img_np = img_nii.get_fdata(dtype=np.float32)
                    if vol["transpose"]:
                        img_np = np.moveaxis(img_np, 0, 1)
                    rn_from, rn_to = vol["range"][0], vol["range"][1]+1                
                    img_np = img_np[rn_from:rn_to]
                    img_np_norm = normalize_intensity(img_np, normalization, percentile)
                    if(len(img_np_norm.shape)==3): #source shape shoud be [x, 180, 180, 1]
                        img_np_norm = np.expand_dims(img_np_norm, -1)
                    assert len(img_np_norm.shape) == 4 , "Incorrect source array shape"            
                    label_p = os.path.join(init_path,  vol["label"])
                    lb_nii = nib.load(label_p)
                    lb_np = lb_nii.get_fdata(dtype=np.float32).astype(np.uint8)
                    if vol["transpose"]:
                        lb_np = np.moveaxis(lb_np, 0, 1)                       
                    lb_np = lb_np[rn_from:rn_to]                     
                    assert np.array_equal(np.unique(lb_np), np.array([0, 1, 2, 3, 4])), "Incorrect label values, expected [0, 1, 2, 3, 4] " + label_p + " " + str(np.unique(lb_np))
                    if(len(lb_np.shape)==4): #target shape shoud be [x, 180, 180]
                        lb_np = np.squeeze(lb_np)
                    assert len(lb_np.shape) == 3 , "Incorrect target array shape"            
                    list_source.append(img_np_norm)
                    list_target.append(lb_np)                
  
    source = np.concatenate(list_source)
    target = np.concatenate(list_target)   
    
    return source, target

def load_single_sample(source_p, label_p, normalization, transpose):
    img_nii = nib.load(source_p)
    #img_nii = nib.as_closest_canonical(img_nii)
    img_np = img_nii.get_fdata(dtype=np.float32)
    if transpose:
        img_np = np.moveaxis(img_np, 0, 1)
    img_np_norm = normalize_intensity(img_np, normalization, percentile=0.8)
    if(len(img_np_norm.shape)==3): #source shape shoud be [x, 180, 180, 1]
        img_np_norm = np.expand_dims(img_np_norm, -1)
    assert len(img_np_norm.shape) == 4 , "Incorrect source array shape"
    lb_nii = nib.load(label_p)
    lb_np = lb_nii.get_fdata(dtype=np.float32).astype(np.uint8)
    if transpose:    
        lb_np = np.moveaxis(lb_np, 0, 1)
    assert np.array_equal(np.unique(lb_np), np.array([0, 1, 2, 3, 4])), "Incorrect label values, expected [0, 1, 2, 3, 4] " + label_p
    if(len(lb_np.shape)==4): #target shape shoud be [x, 180, 180]
        lb_np = np.squeeze(lb_np)
    assert len(lb_np.shape) == 3 , "Incorrect target array shape" 
    
    return img_np_norm, lb_np

def load_single_source(source_p, normalization, transpose):
    img_nii = nib.load(source_p)
    #img_nii = nib.as_closest_canonical(img_nii)
    img_np = img_nii.get_fdata(dtype=np.float32)
    if transpose:
        img_np = np.moveaxis(img_np, 0, 1)
    img_np_norm = normalize_intensity(img_np, normalization, percentile=0.8)
    if(len(img_np_norm.shape)==3): #source shape shoud be [x, 180, 180, 1]
        img_np_norm = np.expand_dims(img_np_norm, -1)
    assert len(img_np_norm.shape) == 4 , "Incorrect source array shape"
    
    return img_np_norm 


def handle_relative_path(cwd, *args):
    paths = []
    for pth in args:
        if pth.startswith(('.', '..')):
            pth = os.path.join(cwd, pth) 
        paths.append(pth)    
    return  paths 

