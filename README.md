### Members Unique Names
- alecsoc
- amakuch
- higginss
- najariac

# EECS556W21_final_project
Final project for EECS 556: Image processing of Winter 2021

The goal of this project is to replicate the algorithm and results of " Double Gaussian mixture model for image segmentation with spatial relationships " by T. Xiong et al. [1] in Julia. 


## Paper Abstract
In this report, we describe several image segmentation models and present our results from the implementation and evaluation of the Double Gaussian mixture model described in [1] for the segmentation of various image sets (natural, synthetic, and medical images). We compare this model's performance to that of a standard GMM model, a standard k-means clustering model, and a convolutional neural network (CNN) segmentation model trained on our medical images data set [2]. Our performance metrics applied to evaluate these methods were the Probabilistic Rand (PR) index and the Dice Score, and we found that the CNN method generally performed much better than the other segmentation models, and that the DGMM performed better than the GMM and k-means for the natural and synthetic images but had poor performance for the medical images.

## Example Results
This first example compares the synthetic image and segmentation results for k-means, gmm, and dgmm from [1].
![Grayscale_comparison](results/Sythetic_comparison.png)

This is an example comparing natural image and segmentation results for k-mean, gmm, and dgmm from [1].
![Natural_comparison](results/natural_comparison.png)


### Description
The code in this repo attempts to reproduce the results found in [1] in julia. The k-means and gaussian mixture model(GMM) are utlized from the following [Clustering](https://juliastats.org/Clustering.jl/) and [GaussianMixtures](https://github.com/davidavdav/GaussianMixtures.jl). The Double gaussian mixture model(DGMM) is implemented in the from scratch in [DGMM](DGMM.jl).

In addition to the implemented DGMM, we tested image segmentation with the medical dataset [2] with a convolutional neural net (CNN). The model framework is derived from a U-net. All testing, model, and data files are located in [CNN](CNN/). The UNet_model.jl code has been modified from previous code [3].

The [performance.jl](performance.jl) is a collection of performance related functions such as misclassification ratio(MCR), the probabalistic rand index(PR), and DICE score.

### Testing Function
The following are testing function used to implement the k-means, GMM, and DGMM with sytheic images, natural images, and medical images.
- [kmean_test.jl](kmean_test.jl)
- [gaussian_mixture_model_test.jl](gaussian_mixture_model_test.jl)
- [DGMM_test_medical.jl](DGMM_test_medical.jl)
- [DGMM_test.jl](DGMM_test.jl)

### Data Collection and Synthesis
The following are files used to collect image, and image data from datasets.
- [dataCollection.jl](dataCollection.jl)
- [medicalDatacollect.jl](medicalDatacollect.jl)
- [partition_medical_image_data.jl](partition_medical_image_data.jl)

Additionally, the sythetic image to replicate from [1] is created via [synthetic_image_creation.jl](sythetic_image_creation.jl) 

### Data and Images
The following are folder with data and images:
- [test_images](test_images)
- [Brain_dataset](Brain_dataset)
- [ground_truth](ground_truth)

Lastly, some of our preliminary results can be found in the results folder. These images in this folder are not necessarily final nor correct.

## References:
[1] T. Xiong, L. Zhang, and Z. Yi. ???Double Gaussian mixture model for image segmentation with spatialrelationships???. In:Journal of Visual Communication and Image Representation34 (2016), 135???145.DOI:10.1016/j.jvcir.2015.10.018

[2] Quantification of Uncertainties in Biomedical Image Quantification Challenge. 2020. URL:https://qubiq.grand-challenge.org/Home/

[3] https://gist.github.com/haampie/bceb1d59fd9a44f092f913062e58d482

