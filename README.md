# Biologically RElevant VIrtual Staining

In this repository we provide the code base that accompanies the paper:

## Learning to see colours: generating biologically relevant fluorescent labels from bright-field images

#### by Håkan Wieslander, Ankit Gupta, Ebba Bergman, Erik Hallström and Philip J Harrison

<p>
    <img src="overview.png" alt="drawing" style="width:1200px;"/>
    <center>Proposed method and workflow. Cell cultures are imaged by fluorescence and bright-field imaging. These images are used to train and compare specialized models for generating each stain, based on biologically relevant features. The selected models are then used to virtually stain bright-field images.</center>
</p>

<p>
    <img src="final_results.png" alt="drawing" style="width:1200px;"/>
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
