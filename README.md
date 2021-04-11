# EECS556W21_final_project
Final project for EECS 556: Image processing of Winter 2021

The goal of this project is to replicate the algorithm and results of " Double Gaussian mixture model for image segmentation with spatial relationships " by T. Xiong et al. [1] in Julia.

File structure: 
The natural images for our preliminary tests of our methods are stored in the test_images folder, and the ground_truth folder contains ground truths for these natural images. Results of segmentation of these images are in the results folder. Our methods that we are comparing our implemented methods to, the standard gaussian mixture model, and the k-means method, are implemented in the kmean_test.jl and gaussian_mixture_model_test.jl files. The synthetic_image_creation.jl file contains the code used to generate synthetic images matching those in the original DGMM paper. The performance.jl file contains the implementations of the MCR and PR metrics for segmentation performance analysis. The gmm_python_test.py file generates figures from our gmm results for the paper. The dataCollection.jl and ground_truth.jl are older files we used earlier on in the project to run the gmm and kmeans analysis on the images and generate figures. Our DGMM model we've developed based on the contents of the paper is in the DGMM.jl file, and the test code we use to test our DGMM model is the DGMM_test.jl file.

## References:
[1] T. Xiong, L. Zhang, and Z. Yi. “Double Gaussian mixture model for image segmentation with spatialrelationships”. In:Journal of Visual Communication and Image Representation34 (2016), 135–145.DOI:10.1016/j.jvcir.2015.10.018
