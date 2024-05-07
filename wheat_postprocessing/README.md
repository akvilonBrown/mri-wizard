# Postprocessing and analysis

The postprocessing of segmented images is performed by several scripts that run one after another; they are accordingly enumerated.
The input data consists of source and segmented images grouped in folders according to probes/cultivar distribution. Number of images in folder are not limited, the distribution should be described in the Excel file. The demo data and description file are provided. These must be downloaded and placed in the **data** folder (check the preferable folder structure in the *README* under **data** folder). It is enough to start postprocessing. The demo data also contains the intermediate and final outputs for inspection - you can remove them to check how postprocessing scripts work step by step and create the debug data in progress, following the setting in the **config** file.

*00_instance_separation.py* script takes the input data as source and segmented images, where the grains are scanned tightly packed, and separates them into individual instances. In the main course of the *MRI Wizard* project, grains were scanned in triplets. The path to the parent folder of data (*pre*) and the path to the data description file (*config_file*) are provided in *config*. The algorithm can determine the number of instances on an image by using clustering, but it is preferable to work with a predefined number of instances (*setting/separation/bestnum*). The dimensions of output images, the folder where to place them, and the description file are also specified in *config*. This file is then used in the next stage. Pericarp tissues can be optionally removed in the separated instances.

*01_analysis_initial.py* performs initial volumetric analysis of images with separated grains. Collecting the required data at this stage is preferable because minor artifacts may appear after further processing due to the interpolation of voxels. The results are saved in a summary file. This script is currently memory-hungry and may require up to 100 GB of RAM. In further versions, optimization will be considered.

*02_pca_first.py* performs quick alignment according to the longest axis of grain. It produces the aligned instances and cross-section through the center of these images. Data description file i. They are used as input for the next step and for visual inspection when debugging the workflow.

*03_rotation_detect.py* checks the central cross-section of images aligned with PCA on the previous step and outputs the angle value that can finish the alignment of grain by rolling around the longest axis (like a lump of grilled meat).

*04_pca_final.py* combines the rotation from PCA with the rolling angle detected in the previous step forming a single rotation matrix applied to the separated grains from step #00. Therefore, images rotated with the *02_pca_first.py* are used only for step #03, but not involved in consecutive rotation in favor of one-step rotation to minimize artifact emergence. 

*05_crease_depth.py* script calculates the height of the fold arc on central cross-sections of aligned images. Debug images are saved to check plausibility.

*06_analysis_bbox.py* measures the linear dimensions of aligned grains.

*07_finalizing.py* stitches all data together and brings values into millimeter format based on the resolution, provided in *config*
