using Flux
using Flux: @epochs
using Plots
using FileIO: load
using BSON: @save
using BSON: @load
using Zygote
using MIRTjim:jim

##########
# Load data
##########


Image_path = "test_images\\"

train_set_image = load(Image_path * "Train_Set_Images.jld2", "train_set_image")
train_set_label = load(Image_path * "Train_Set_Labels.jld2", "train_set_label")
test_set_image = load(Image_path * "Test_Set_Images.jld2", "test_set_image")
test_set_label = load(Image_path * "Test_Set_Label.jld2", "test_set_label")

train_dataset = Flux.Data.DataLoader((train_set_image,train_set_label), batchsize = 4, shuffle = true)
test_dataset = Flux.Data.DataLoader((test_set_image,test_set_label), batchsize = 2, shuffle = true, partial = false)

#@load "CNN_model.bson" model_unet


##########
# Set up and train model
##########


include("UNet_Model.jl")
model_unet = Create_Modified_Unet_Model()

function loss(x, y)
         Flux.binarycrossentropy(model_unet(x),y)
end

test_x, test_y = (test_set_image[:,:,:,1,:],test_set_label[:,:,:,1,:])

function evalcb()
        @show(loss(test_x, test_y))
        @save "CNN_model.bson" model_unet
end

@epochs 30 Flux.train!(loss, Flux.params(model_unet), train_dataset, ADAM(), cb = evalcb)


##########
# View Images and compute accuracies
##########

include(pwd()*"\\performance.jl")

viewimage = 5

p1 = jim(test_set_image[:,:,:,viewimage]);
p2 = jim(test_set_label[:,:,:,viewimage]);
p3 = jim(model_unet(test_set_image[:,:,:,viewimage,:]).>0.5);

plot(p1,p2,p3)
@show DICE(model_unet(test_set_image[:,:,:,viewimage,:]).>0.5,test_set_label[:,:,:,viewimage])

train_dice = 0
for i in 1:size(train_set_image,4)
        dice = DICE(model_unet(train_set_image[:,:,:,i,:]).>0.5,train_set_label[:,:,:,i])
        global train_dice = train_dice + dice
end

train_dice = train_dice / size(train_set_image,4)
@show train_dice

test_dice = 0
for i in 1:size(test_set_image,4)
        dice = DICE(model_unet(test_set_image[:,:,:,i,:]).>0.5,test_set_label[:,:,:,i])
        global test_dice = test_dice + dice
end
test_dice = test_dice / size(test_set_image,4)
@show test_dice


##########
# Determine number of parameters
##########

paramnum = 0
unet_params = Flux.params(model_unet)
for i in 1:length(unet_params)
        paramnum = paramnum + length(unet_params[i])
end
