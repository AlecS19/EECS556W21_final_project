#DGMM data collection
#- TEST CODE FOR MED DATASET - JLD2 -#
using Images: load, channelview
include("performance.jl")
include("DGMM.jl")
using MIRT: jim
using Plots
using Images
using LinearAlgebra
using DelimitedFiles
using Statistics: median
#load medical dataset
picnum = "Test_Set_images"
filepath1 = picnum *".jld2"
testset = load(filepath1)["test_set_image"] #7 images in form [:,:,1,i]
m = size(testset,1); n=size(testset,2)
picnum = "Test_Set_Label"
filepath1 = picnum *".jld2"
GTset = load(filepath1)["test_set_label"]
#define dice score array
dice = zeros(7)
#number of segs
numsegs = 3
numiters = 3
#mkdir("results/DGMM_med/customTest5med/")
for i in 1:7
    #test = mapwindow(median,testset[:,:,1,i],(5,5))
    test = testset[:,:,1,i]
    test_flat = Array{Float64,2}(reshape(test, 1, :)')
    A = DGMM(test_flat, numsegs, (m, n))
    E = log_lik(A, test_flat)
    gray_history = train(A, test_flat, numsegs, numiters)
    p1 = plot(gray_history)
    display(p1)
    D = post_prob(A)
    ass = [argmax(D[i, :]) for i = 1:size(D, 1)]
    segmented_gmm = reshape(ass, ( m,n ) )
    
    #segmented_gmm = (segmented_gmm) .-1
    #segmented_gmm = segmented_gmm .== 3 #flatten the segmentations
if segmented_gmm[1,1] == 1 #check what background is classified as
    a=2
    b=3
elseif segmented_gmm[1,1] == 2
    a=1
    b=3

else
    a=1
    b=2
end
(_,corrseg) = findmax([sum(segmented_gmm .== a),sum(segmented_gmm .== b)]) #find which class has most pixels

if corrseg == 1 #set segmented_gmm to class with most pixels
    segmented_gmm = segmented_gmm .== a
else
    segmented_gmm = segmented_gmm .== b
end

    GT=GTset[:,:,1,i]
    intersection = (segmented_gmm .+ GT) .== 2
    d = 2*sum(intersection) / (sum(segmented_gmm) + sum(GT))
    dice[i] = d
    save("results/DGMM_med/customTest5med/test_set_images_$i .png", colorview(Gray, segmented_gmm'))
end
writedlm("results/DGMM_med/customTest5med/dice.txt", dice)


using Flux
using Flux: @epochs
using Plots
using FileIO: load
using BSON: @save
using BSON: @load
using Zygote
using MIRT

##########
# Load data
##########


#Image_path = "Brain_dataset\\"
Image_path = ""

train_set_image = load(Image_path * "Train_Set_Images.jld2", "train_set_image")
train_set_label = load(Image_path * "Train_Set_Labels.jld2", "train_set_label")
test_set_image = load(Image_path * "Test_Set_Images.jld2", "test_set_image")
test_set_label = load(Image_path * "Test_Set_Label.jld2", "test_set_label")

train_dataset = Flux.Data.DataLoader((train_set_image,train_set_label), batchsize = 4, shuffle = true)
test_dataset = Flux.Data.DataLoader((test_set_image,test_set_label), batchsize = 2, shuffle = true, partial = false)
include("UNet_Model.jl")
@load "CNN_model.bson" model_unet

##########
# View Images and compute accuracies
##########


include("performance.jl")
for i in 1:7
    viewimage =i

p3 = jim(model_unet(test_set_image[:,:,:,i,:]).>0.5)
savefig("results/CNN/test_set_images_$i _CNN.png")

@show dice_cnn[i] = DICE(model_unet(test_set_image[:,:,:,viewimage,:]).>0.5,test_set_label[:,:,:,viewimage])


end
