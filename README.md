# Investigating the efficacy of classical machine learning algorithms for the automated identification of artistic movements

This is the implementation of my [final year project](https://drive.google.com/file/d/1AKLcAOen1EjgpSDOYlnB4z1WNUfpUbS8/view?usp=sharing), which compared the efficacy of different novel classical machine learning models against convolutional neural networks implemented in [this ICIP](http://web.fsktm.um.edu.my/~cschan/doc/ICIP2016.pdf) paper.

## Prerequisites
You will require python 3.6.7 and the following python packages
* Scikit-Learn (0.20.2)
* Scikit-Image (0.15.0)
* Numpy (1.16.3)

These can all be easily installed with pip, or any other python package manager. For pip the installation for a given package is as follows:
```
pip install packagename
```

### Quick Start
If you just wish to test out the best performing model clone or download "*Single Image DAISY Forest*" run "SingleImageTest" and follow the on screen instructions.
Two images are provided for testing purposes. 

To fully recreate this project you will need the build scripts as well as the additional files. For a full guide on how to recreate this project please refer to the "user guide" section of the [report](https://drive.google.com/file/d/1AKLcAOen1EjgpSDOYlnB4z1WNUfpUbS8/view?usp=sharing).
```bash
├── Build
│   ├── Ext								~Scripts Regarding Extracting Feature Data
│   ├── Models
│   │   ├── DAISY
│   │   │   ├── Machines						~Scripts to generate 20 DAISY classifiers for each model
│   │   │   └── Training						~Scripts to train DAISY models
│   │   └── ORB
│   │├── Machines							~Scripts to generate 20 ORB classifiers for each model
│   │└── Training							~Scripts to train ORB models
│   └── Norm								~Scripts used in normalization
└── Testing								~Scripts used to test models against all data
```
### Additional Files
If you wish to recreate this project from scratch you will need the [wikiart dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) (Size = ~26Gb).

If you would instead prefer the extracted feature data with the pre-trained models [you will need this](https://drive.google.com/open?id=1OC_psStovltRR9P-Td2NvY8AUwm0FBcV) (Size = ~1Gb).
