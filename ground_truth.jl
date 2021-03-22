using Images
using Clustering
using MIRT: jim
using Plots
using MAT


#Load Test Image
curr_dir = pwd()
filename1 = "227092.jpg"
filepath1 = curr_dir * "/EECS556W21_final_project/test_images/" * filename1

#Load Ground Truths
filename2 = "227092.mat"
filepath2 = curr_dir * "/EECS556W21_final_project/ground_truth/" * filename2

file = matopen(filepath2)
correct = read(file, "groundTruth") # note that this does NOT introduce a variable ``varname`` into scope
close(file)

#Convert image data to array data [0-255]
test = channelview(load(filepath1))
c, m,n = size(test)

#Plot correct imported image
plot1 = jim( test.*255,title="Original" )
display(plot1)

#Flatten and run k-means based on intensity
test_flat = [ vec( test[1,:,:] ) vec( test[2,:,:] ) vec( test[3,:,:] ) ]'
results = kmeans( test_flat, 3;maxiter = 200,display=:iter)

segmented = results.assignments

#Format it back into an image
segmented = reshape( segmented, ( m,n ) )' .*(255/3)

#plot the outcome
plot2 = jim( segmented, title="segmented")
display(plot2)
## tet

include("performance.jl")
#Display the MIRT
h =  PR(segmented,correct)
