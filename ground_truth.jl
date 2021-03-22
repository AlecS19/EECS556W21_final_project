using Images
using Clustering
using MIRT: jim
using Plots
using MAT


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

#Flatten and run k-means based on intensity
test_flat = [ vec( test[1,:,:] ) vec( test[2,:,:] ) vec( test[3,:,:] ) ]'
results = kmeans( test_flat, 2;maxiter = 200,display=:iter)

segmented = results.assignments

#Format it back into an image
segmented = reshape( segmented, ( m,n ) ) .*(255/3)

#plot the outcome
plot2 = jim( segmented', title="segmented")
display(plot2)
## the

include("performance.jl")
#Display the MIRT
h =  PR_randindex(segmented,correct)
