# MRI Wizard for seeds
This repository contains models involved for initial development of grain segmentaion project, as well as postprocessing scripts and some benchmarks.
The demo data and the main segmentation model should be [downloaded separately](https://ipk-cloud.ipk-gatersleben.de/s/5exapoJ7dbnzcTf), if required.

The folder **xp_pilot_bench** contains 2D U-net segmentation scripts for demo training and inference processing slices of 3D NMR images, in the subfolders created by Hydra configuration tool there is a result of training and checkpoints.
The folder **xp_stacked_model_bench** contains 2.5D U-net segmentation scripts that process stacks of slices.
These models were used on initial stage of the project when annotated data was scarce.
**xp_pilot_bench** and **xp_stacked_model_bench** have **config** subfolder with hyperparameters used for the successful training. With *train.py* script and demo data you can see how it works. Each launch of any script creates a subfolder with logs, checkpoints and debug information, governed by [*Hydra configuration management framework*](https://hydra.cc/). Upon training, a quick test is performed. It can be performed separately with *metrics.py* script when respective checkpoint is specified in *config*. *predict.py* script segments sample with specified trained model checkpoint and folder where to pick up source data.

The more advanced model is *nnUnet*, a 3D segmentation framework, it should be installed according to [official guidelines (version 1.0)](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1/), including setting the environmental variables.
Checkpoints of the trained model should be [downloaded - nnUNet.zip](https://ipk-cloud.ipk-gatersleben.de/s/5exapoJ7dbnzcTf) and unpacked in the folder specified as *RESULTS_FOLDER* in environment variables. 

The folder **rotation_detect** contains auxiliary *ResNet* model which determines the rotation angle of the central slice of a grain, used to finish aligning of a grain in a 3D image. There is script for training with demo data (data should be downloaded - please refer to *data* subfolder with description). There is also a checkpoint of this model which is used in postprocessing scripts (*03_rotation_detect.py*).

The folder **evaluation** contains script to compare segmented images with ground truth and calculate the score.

The folder **wheat_postprocessing** contains script for analazing segmented data. They should be launched one by one in a sequence, guided by configuration in the Hydra file.
Demo data is also available. It can used with a starting files only, but also contains all intermediate files produced by each script for overview. 
In postprocessing another neural network was involved for determining the final rotation of seed sample. To show how it was developed and how it works, the respective code was placed in the folder rotation_detect (demo data available as well).

The folder **wheat_preprocessing** contains script to prepare NMR image for *nnUnet* inference. 

The folder **surface_area_benchmark** contains code that compares the accuracy of surface area calculation in 3D figures with predefined area and volume.

The **data** folder is created for the demo data that can be downloaded and unpacked after cloning this repository on the local computer.

The folders **dataloader_nmr**, **inference**, **models**, **trainer** are local packages providing classes and utilities for the rest of scripts.

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7`. 
GPU is highly recommended for segmentation models as they were tested only with GPU present. Postprocessing scripts can perform without GPU.
You can create virtual environment and install libraties with *pip install* or *conda install* commands. 
Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch dependencies. The other few libraries are listed in the file `requirements.txt`, so they can be installed in bulk by 
```sh
git clone https://github.com/akvilonBrown/mri-wizard
pip install -r requirements.txt
pip install -e .
```
The last command is required for local dependencies to work across project folders and subfolders (if you preferred conda, you have to use *pip install -e* anyway whithin virtual environment to make local packages available).
Jupyter Notebook is optional, it is not required as all scripts can be run in a console.

*nnUnet* can be installed in the same environment, it requires only *Pytorch* as a dependency, which is present in *MRI Wizard* dependencies (the order doesn't matter, you can install *nnUnet* prior to *MRI Wizard*).
According to [official guidelines (version 1.0)](https://github.com/MIC-DKFZ/nnUNet/tree/) after allocating the folder for nnUnet installation you need to perform the following commands:

```sh
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
git checkout nnunetv1
pip install -e .
```
then allocate three service folders and set [environment variables](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/setting_up_paths.md) to use *nnUnet* seamlessly.
Demo data is located in another resource, it can downloaed and description of *nnUnet* usage is placed in the **data** subfolder