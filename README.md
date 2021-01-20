# Biologically RElevant VIrtual Staining

In this repository we provide the code base that accompanies the paper:

https://www.biorxiv.org/content/10.1101/2021.01.18.427121v1

## Learning to see colours: generating biologically relevant fluorescent labels from bright-field images

#### by Håkan Wieslander, Ankit Gupta, Ebba Bergman, Erik Hallström and Philip J Harrison

<p>
    <img src="readme_images/overview.png" alt="drawing" style="width:1200px;"/>
    <center>Proposed method and workflow. Cell cultures are imaged by fluorescence and bright-field imaging. These images are used to train and compare specialized models for generating each stain, based on biologically relevant features. The selected models are then used to virtually stain bright-field images.</center>
</p>

<p>
    <img src="readme_images/final_results.png" alt="drawing" style="width:1200px;"/>
    <center>Comparison of images generated from the bright-field z-stack and the ground truth fluorescence images. (A-C) Generated images for the nuclei, lipid droplets and cytoplasm, with zoomed in regions showing some well reconstructed areas alongside some more problematic locations. (D-F) Ground truth images with corresponding zoomed in regions. (G) Maximum projection of the bright-field z-stack with corresponding zoomed in regions.</center>
</p>

The content and structure of the repo is given by the following: 

```sh
.
├── README.md
├── ai_haste
│   ├── data                : python scripts for loading the datasets required
│   ├── loss                : for loss functions not included in PyTorch
│   ├── model               : python scripts for the various neural networks
│   ├── test                : scripts requied when running the models in test mode
│   ├── train               : scripts requied when running the models in train mode
│   └── utils               : utility functions (e.g. for converting the images to numpy arrays for faster data loading)
│
├── config                  : .json files for running the models to reconstruct the three fluorescence channels
│                       
└── exp_stats               : .csv files for image statistics and train/test splits 
    
```

The data should be structured as:
```sh
.
|--"adipocyte_data"       :Base folder where images are stored
|   |-- "60x images"    :What magnification the images are images at
```


## Data

The image files in the magnification folder are named as ex AssayPlate_Greiner_#655090_{well}_T0001F0{FOV}L01A0{Action_nr}Z0{stack_nr}C0{channel}.tif where "well" and "FOV" are imported from exp_stats. "Action_nr" range from 1-4, stack_nr range from 1-7 for bright-field images and channel is one of C01 (Nuclei), C02 (Lipid droplets), C03 (Cytoplasm), C04 (Bright-field)

## Training

To train the models, modify the content in the config file that you want to run. For instance change the path to the data under "data" - "folder" and the training validation and test splits csv files under "data" -- ""train_csv_file", "validation_csv_file" and "test_csv_file". You can also specify other things such as model configurations, optimizers, learning rate ...

To start training execute for instance for nuclei channel
```sh
python3 -m brevis -c configs/nuclei_train.json 
```

