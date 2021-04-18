"""
#- TEST CODE FOR PNG/JPG/TIF -#
using Images: load, channelview
include("performance.jl")
include("DGMM.jl")
using MIRT: jim
using Plots

picnum = "ID_0004"
filepath1 = "medImage1/"* picnum *".png"
test = convert(Array{Float64}, load(filepath1)) .* 255
m, n = size(test)
test_flat = Array{Float64,2}(reshape(test, 1, :)')
A = DGMM(test_flat, 2, (m, n))
E = log_lik(A, test_flat)
gray_history = train(A, test_flat, 2, 5)
p1 = plot(gray_history)
display(p1)
D = post_prob(A)
ass = [argmax(D[i, :]) for i = 1:size(D, 1)]
segmented_gmm = reshape(ass, ( m,n ) )
jim(segmented_gmm')

"""

#- TEST CODE FOR MED DATASET - JLD2 -#
using Images: load, channelview
include("performance.jl")
include("DGMM.jl")
using MIRT: jim
using Plots
using Statistics
using ImageFiltering

picnum = "Test_Set_Images"
filepath1 = picnum *".jld2"
numsegs = 3
test = load(filepath1)["test_set_image"][:,:,1,2]
test = mapwindow(median, test, (5,5))
m, n = size(test)
test_flat = Array{Float64,2}(reshape(test, 1, :)').*(255/1000) .+ 1
A = DGMM(test_flat, numsegs, (m, n))
E = log_lik(A, test_flat)
gray_history = train(A, test_flat, numsegs, 4)
p1 = plot(gray_history)
display(p1)
D = post_prob(A)
ass = [argmax(D[i, :]) for i = 1:size(D, 1)]
segmented_gmm = reshape(ass, ( m,n ) )
r = jim(segmented_gmm)

display(r)

segmented_gmm = (segmented_gmm) .-1
segmented_gmm = segmented_gmm .> 0 #flatten the segmentations

#GT = (abs.(load("medImage1_GT/ID_0006_AGE_0075_CONTRAST_1_CT.png")) ./ 3)'
picnum = "Test_Set_Label"
filepath1 = picnum *".jld2"
GT = load(filepath1)["test_set_label"][:,:,1,2]
intersection = (segmented_gmm .+ GT) .== 2
a = 2*sum(intersection) / (sum(segmented_gmm) + sum(GT))
#test inverted
segmented_gmm_inverted = segmented_gmm .== 0
intersection = (segmented_gmm_inverted .+ GT) .== 2
b = 2*sum(intersection) / (sum(segmented_gmm_inverted) + sum(GT))

if a > b
    print("Non-inverted Dice score:")
    print(a)
    plot(jim(segmented_gmm),jim(GT))
else
    print("Inverted Dice score:")
    print(b)
    plot(jim(segmented_gmm_inverted),jim(GT))
end
