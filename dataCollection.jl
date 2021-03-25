using Images
using Clustering
using MIRT: jim
using Plots
using MAT
using GaussianMixtures
using LinearAlgebra:I
using Statistics: mean
include("performance.jl")
function kmeanImp(imagename, imagepath, gtname, gtpath, numSegs::Int)
    """
    Inputs:
    test_image (RGB) path, ground_truths (2D Dict) path
    

    Outputs: 
    PR Scores
    MCR Number
    """
    
    #Load Test Image
    curr_dir = pwd()
    filepath1 = curr_dir * imagepath * imagename

    #Load Ground Truths
    filepath2 = curr_dir * gtpath * gtname
    
    file = matopen(filepath2)
    correct = read(file, "groundTruth") # note that this does NOT introduce a variable ``varname`` into scope
    close(file)

    test = channelview(load(filepath1))
    c,m,n = size(test)
    
    
    """###average of Ground Truths
    z = length(correct)
    correctAv = zeros(size(correct[1]["Segmentation"]))
    for k = 1:z
        correctAv += correct[k]["Segmentation"]
    end
    correctAv = Array{Float64}(correctAv ./ z)
    """

    ## kmeans
    #Flatten and run k-means based on intensity
    test_flat = [  vec( test[1,:,:] ) vec( test[2,:,:] ) vec( test[3,:,:])  ]'.*256
    #For some reason scaling to a 255 allows correct segmentation of the skier
    results = kmeans( test_flat, numSegs;init=:rand,tol=1e-6,display=:iter)
    #run kmeans twice to get correct image? Maybe initialization is bad
    #init set to rand seemed to fix the issue
    
    segmented = results.assignments
    segmented = reshape(segmented,(m,n)) .*(255/3)
    
    PR_kmean = PR_fast(segmented, correct)
    #x1MCR_kmean = MCR(segmented, correctAv)
    
    z = length(correct)
    MCRs = zeros(z)
    for i = 1:z
        MCRs[i] = MCR(segmented, Array{Float64}(correct[i]["Segmentation"]))
    end
    
    MCR_kmean = mean(MCRs)
    
    
    return PR_kmean, MCR_kmean
end

function gmmImp(imagename, imagepath, gtname, gtpath, numSegs::Int)
    
    #Load Test Image
    curr_dir = pwd()
    filepath1 = curr_dir * imagepath * imagename

    #Load Ground Truths
    filepath2 = curr_dir * gtpath * gtname
    
    file = matopen(filepath2)
    correct = read(file, "groundTruth") # note that this does NOT introduce a variable ``varname`` into scope
    close(file)

    test = channelview(load(filepath1))
    c,m,n = size(test)
    
    test_flat = [  vec( test[1,:,:] ) vec( test[2,:,:] ) vec( test[3,:,:])  ]'.*256
    data_gmm = Array{Float64,2}(test_flat')

    gm = GMM(numSegs,data_gmm;nInit=50,kind=:diag)#full give better accuracy, but not the same picture, when you add extra iterations
    #This seems to give the DGMM results in terms of picture quality and PR ???

    #gm.μ = results.centers'
    #gm.μ = [100 100 100; 200 200 200 ]

    GaussianMixtures.em!(gm,data_gmm;nIter=200)
    prob = GaussianMixtures.gmmposterior(gm, data_gmm)[1]
    ass=[argmax(prob[i,:]) for i=1:size(data_gmm,1)]
    segmented_gmm = reshape( ass, ( m,n ) ) .*(255/3)
    
    PR_gmm = PR_fast(segmented_gmm,correct)
    
    
    
    z = length(correct)
    MCRs = zeros(z)
    for i = 1:z
        MCRs[i] = MCR(segmented_gmm, Array{Float64}(correct[i]["Segmentation"]))
    end
    
    MCR_gmm = mean(MCRs)
    
    return PR_gmm, MCR_gmm
    
end

#example:
#gmmImp("260058.jpg", "/test_images/", "260058.mat", "/ground_truth/",2)
    


