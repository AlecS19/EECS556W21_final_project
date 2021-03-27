using Images
using Clustering
using MIRT: jim
using Plots
include("performance.jl")

#Load in figure 4
curr_dir = pwd()
filename1 = "figure4b.png"
filename2 ="figure4a.png"

filepath1 = curr_dir * "/EECS556W21_final_project/test_images/" * filename1
filepath2 = curr_dir * "/EECS556W21_final_project/test_images/" * filename2

#Convert image data to array data [0-255]
test = convert(Array{Float64}, load(filepath1) )*255
correct = convert(Array{Float64}, load(filepath2) )'*255

#Plot correct imported image
plot1 = jim( correct,title="Correct" )
display(plot1)

#Flatten and run k-means based on intensity
test_flat = reshape(test,1,:)
results = kmeans( test_flat, 4)

#Sort the categories to match the intensity categoies of the original image
val = [0, 80, 170, 255]
t = sortperm(results.centers[:]; rev=false)
p = val[ sortperm(results.centers[:]; rev=true) ]

segmented = results.assignments
for i in 1:4
    segmented[segmented .== i] .= val[t[i]]
end

#Format it back into an image
segmented = reshape( segmented, (128,128) )'

#plot the outcome
plot2 = jim( segmented, title="segmented")
display(plot2)

#Display the MIRT
print( MCR(segmented,correct) )
displayable = Array{Float64,2}(segmented'./255)
save("figure4_segmented_kmeans.png", colorview(Gray, displayable))
