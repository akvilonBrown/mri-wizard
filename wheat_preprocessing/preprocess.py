import numpy as np
import nibabel as nib
import os
from sklearn.preprocessing import minmax_scale


'''
nnUnet was trained on the NMR images with an intensity range from 0 to 32000 
and with 4D shape like [x, 180, 180, 1], where the last dimension is just an expansion of the 3rd.
The header information was not used and cleared therefore to ensure all files are treated equally.
The resolution value is used afterwards in postprocessing scripts separately to calculate features based on real dimensions.
Also renaming and packing can be required, since nnUnet uses gzip files with naming convention having a modality in the file name,
like file0000.nii.gz, where 0000 is the only modality used. 

'''

def main():
    source_pt = "../path/file.nii" 
    source_save = "../path/file0000.nii.gz"
    intensity_range = (0,32000)

    img_nii = nib.load(source_pt)
    img_npt = img_nii.get_fdata(dtype=np.float32)    
    print(img_nii.header)
    img_npt = img_npt.astype(np.int16)
    print(f"{img_npt.shape = }, {img_npt.min()= }, {img_npt.max()= }")

    ## Resize and rescale,. remove pixdim and pack
        
    if(len(img_npt.shape)==3): #source shape shoud be [x, 180, 180, 1]
        img_npt = np.expand_dims(img_npt, -1)          

    shape = img_npt.shape
    img_np = minmax_scale(img_npt.ravel(), feature_range=intensity_range).reshape(shape)
    img_np = img_np.astype(np.int16)
    print(f"{img_np.shape = }, {img_np.min() = }, {img_np.max()= }")
    img_nifty = nib.Nifti1Image(img_np, affine=np.eye(4))

    #pixdim = np.array([1., 0.04, 0.0401786, 0.0401786, 0., 0., 0., 0.]) #  unused 
    #img_nifty.header['pixdim'] = pixdim
    #print(img_nifty.header)    

    nib.save(img_nifty, source_save) 
    print("Completed")

if __name__ == '__main__': 
        main()


 


