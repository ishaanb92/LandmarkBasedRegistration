# Landmark-guided deformable image registration 

We study the influence of (learned) landmark correspondences on intensity-based deformable image registration involving "hard" organs like the lung and liver. Our work extends the self-supervised model developed by [Grewal et al. (2023)](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-10/issue-01/014007/Automatic-landmark-correspondence-detection-in-medical-images-with-an-application/10.1117/1.JMI.10.1.014007.full#_=_) by proposing the use of a mask during training to focus the model on key anatomical structures (e.g. vessels inside the liver). 

We demonstrate the benefits our *soft mask* extension using two use-cases:
* Lung CT registration (using the 4DCT and COPDgene datasets)
* Liver lesion co-localization (using a dataset of dynamic contrast-enhanched (DCE) MR images collected at UMC Utrecht)


# Model

![Landmark correspondence prediction model](landmark_model.png)

# Results

### Lung CT registration
[!Lung CT registration](copd_results.png)

### Liver lesion co-localization
[!Liver lesion co-localization](lesion_matching.png)

# Usage

Use the following to clone the repository and install packages.

    git clone https://github.com/ishaanb92/LandmarkBasedRegistration.git
    python setup.py install

You will also need to install Elastix yourself from [here](http://elastix.lumc.nl/download.php). Set the `elastix_path` and `transformix_path` to the paths where you installed the binaries for `elastix` and `transformix` when using the `ElastixInterface` and `TransformixInterface` classes during registration. 

## Citation
Our paper is currently under peer-review. This section will be updated after publication. 

## Useful links
* [Elastix manual](https://elastix.lumc.nl/download/elastix-5.1.0-manual.pdf)
* [Using gryds to deform images](https://github.com/tueimage/gryds/blob/master/notebooks/tutorial.ipynb)
* [Why do we need landmarks?](https://iopscience.iop.org/article/10.1088/0266-5611/24/3/034008)


