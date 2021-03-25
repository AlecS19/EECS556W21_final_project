using Images
using Clustering
using MIRT: jim
using Plots
using MAT
using GaussianMixtures
using LinearAlgebra:I

numSegs = 2
#Load Test Image
curr_dir = pwd()
filename1 = "260058.jpg"
filepath1 = curr_dir * "/EECS556W21_final_project/test_images/" * filename1

#Load Ground Truths
filename2 = "260058.mat"
filepath2 = curr_dir * "/EECS556W21_final_project/ground_truth/" * filename2

file = matopen(filepath2)
correct = read(file, "groundTruth") # note that this does NOT introduce a variable ``varname`` into scope
close(file)

#Convert image data to array data [0-255]
test = channelview(load(filepath1))
c, m,n = size(test)

## kmeans
#Flatten and run k-means based on intensity
test_flat = [  vec( test[1,:,:] ) vec( test[2,:,:] ) vec( test[3,:,:])  ]'.*256
#For some reason scaling to a 255 allows correct segmentation of the skier
results = kmeans( test_flat, numSegs;init=:rand,tol=1e-6,display=:iter)
#run kmeans twice to get correct image? Maybe initialization is bad
#init set to rand seemed to fix the issue

segmented = results.assignments

#Format it back into an image
segmented = reshape( segmented, ( m,n ) ) .*(255/3)

#plot the outcome
plot2 = jim( segmented', title="segmented kmeans")
display(plot2)

## gmm
data_gmm = Array{Float64,2}(test_flat')

gm = GMM(numSegs,data_gmm;nInit=50,kind=:diag)#full give better accuracy, but not the same picture, when you add extra iterations
#This seems to give the DGMM results in terms of picture quality and PR ???

#gm.μ = results.centers'
#gm.μ = [100 100 100; 200 200 200 ]

GaussianMixtures.em!(gm,data_gmm;nIter=200)
prob = GaussianMixtures.gmmposterior(gm, data_gmm)[1]
ass=[argmax(prob[i,:]) for i=1:size(data_gmm,1)]
segmented_gmm = reshape( ass, ( m,n ) ) .*(255/3)

plot3 = jim(segmented_gmm',"gmm segmented")
display(plot3)
## the

include("performance.jl")
#Display the MIRT
PR_kmeans =  PR_fast(segmented,correct)
PR_gmm = PR_fast(segmented_gmm,correct
