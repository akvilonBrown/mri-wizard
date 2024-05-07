# Data comes here

Placeholder for the demo data (should be downloaded separately, check [**data** archive](https://ipk-cloud.ipk-gatersleben.de/s/5exapoJ7dbnzcTf)) 
The archive contains data for different models in this project, it's optional to select only those subfolder which you are actually going to use.
After unpacking the structure of this folder should look the following (not mandatory, but it fits the paths in configuration paths in other parts of the project)

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
├── rotation_detect_data│   
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


