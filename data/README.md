# Data comes here

Placeholder for the demo data (should be downloaded separately; check [**data** archive](https://ipk-cloud.ipk-gatersleben.de/s/5exapoJ7dbnzcTf)) 
The archive contains data for different models in this project; it's optional to select only those subfolders you will use.
After unpacking, the structure of this folder should look the following (not mandatory, but it fits the paths in configuration paths in other parts of the project)

```
data
├── demo_data_for_2D_models
│   ├── prediction_nnunet
│   ├── prediction_pilot_model
│   ├── prediction_stacked_model
│   ├── test
│   │   ├── label
│   │   └── source
│   ├── train
│   └── validation
├── demo_data_nnunet
├── rotation_detect_data   
│   └── rotation_dataset
└── segmented_probes    
    ├── probe1
    ├── probe2
    ├── probe3
    └── analysis 
        ├── _aligned
        │   ├──pca_first
        │   │  ├──debug
        │   │  └──data
        │   │     ├── probe1
        │   │     ├── probe2
        │   │     └── probe3
        │   └──pca_final
        │      ├──crease
        │      ├──debug
        │      └──data
        │         ├── probe1
        │         ├── probe2
        │         └── probe3               
        │
        └── _separated
            ├── probe1
            ├── probe2
            └── probe3
```

In the **demo_data_for_2D_models** folder, there is training, test, and validation data for 2D and 2.5D models (described in the *dataloader.json* file used and additional config for models). Training data consists of 18 NMR images from the middle stage of the project. The predictionXX folders are empty - they are placeholders where you can forward the output of the models in the inference mode (script *predict.py in the respective project folders*) in order to use this data for evaluation.
The folder **demo_data_nnunet** contains three preprocessed source NMR images for inference by *nnUnet*. When your *nnUnet* framework is installed (references to the steps on the main page), the checkpoints from the latest training can be downloaded and placed in the folder specified as *RESULTS_FOLDER* in environment variables. Then, you can segment probes with a command:
```sh
nnUNet_predict -i ./data/demo_data_nnunet -o ./data/demo_data_nnunet -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task580_WheatBarley
```
The segmented probes can be used for analysis. However, the segmented probes are already included in the demo data for postprocessing if you want to skip segmentation.
The command for segmenting test data
```sh
nnUNet_predict -i ./data/demo_data_for_2D_models/test/source -o ./data/demo_data_for_2D_models/prediction_nnunet -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task580_WheatBarley
```
Please remove files *postprocessing*, *plans*, and *prediction_time* and rename the prediction files if required to match the file names of ground truth files when evaluation is performed (evaluation script requires file names to match when comparing predicted files and ground truth)

The **rotation_detect_data** folder contains the dataset for training the *ResNet* rotation detection model and dataloader configuration file.
The folder **segmented_probes** contains all data used for and obtained in postprocessing, including intermediate steps. If you are going to perform postprocessing, you may remove the **analysis** subfolder and start with probes data and data description file datalist_probe.xlsx. All subsequent structures will be recreated according to the postprocessing configurations. The detailed description is in the **wheat_postprocessing** folder.
