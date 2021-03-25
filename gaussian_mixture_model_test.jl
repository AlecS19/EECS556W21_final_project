using Plots
using Images
using MIRT: jim


include("performance.jl")

#Load in figure 4
curr_dir = pwd()
filename1 = "figure4b.png"
filename2 ="figure4a.png"

filepath1 = curr_dir * "/EECS556W21_final_project/test_images/" * filename1
filepath2 = curr_dir * "/EECS556W21_final_project/test_images/" * filename2

#Convert image data to array data [0-255]
test = convert(Array{Float64}, load(filepath1) )'*255
correct = convert(Array{Float64}, load(filepath2) )'*255

test_flat = vec(test)
correct_flat = vec(correct)

htest = histogram(test_flat,bins=0:255)
hcor = histogram(correct_flat,bins=0:255)

p1 = plot(htest, title="Noisy Image fig 4")
p2 = plot(hcor, title="Original Image fig 4")

## Testing Gaussian Mixtures
#Gaussian Mixture Model for fig 4
using GaussianMixtures

iris = dataset("datasets", "iris")
classes = unique(iris[:,5])
#Must be a size of Array{?,2}
h = reshape(test, (128*128,1) )

gm  = GMM(4,h)

results = GaussianMixtures.em!(gm,h)
prob = GaussianMixtures.gmmposterior(gm, h)[1]
ass=[argmax(prob[i,:]) for i=1:size(h,1)]

#Sort the categories to match the intensity categoies of the original image
centers = gm.μ

val = [0,80,170,255]
t = sortperm(centers[:])

segmented = ass
for i in 1:4
    segmented[segmented .== i] .= val[ findall(x->x==i,t) ]
end
segmented = reshape( segmented, (128,128) )
jim(segmented)

print(MCR(segmented,correct))
