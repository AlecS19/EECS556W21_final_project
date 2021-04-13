using Flux
using Flux: @epochs
using Plots
using FileIO: load
include("UNet_Model.jl")
model_unet = Create_Modified_Unet_Model()

load("medical_dataset\\Train_Set_Images.jld2", "train_set_image")
load("medical_dataset\\Train_Set_Labels.jld2", "train_set_label")
load("medical_dataset\\Test_Set_Images.jld2", "test_set_image")
load("medical_dataset\\Test_Set_Label.jld2", "test_set_label")


train_dataset = Flux.Data.DataLoader((train_set_image,train_set_label), batchsize = 16, shuffle = true)

test_dataset = Flux.Data.DataLoader((test_set_image,test_set_label), batchsize = 4, shuffle = true, partial = false)

function loss(x, y)
         Flux.binarycrossentropy(model_unet(x),y)
 end

test_x, test_y = (test_set_image[:,:,:,1,:],test_set_label[:,:,:,1,:])#
evalcb() = @show(loss(test_x, test_y))

@epochs 10 Flux.train!(loss, Flux.params(model_unet), train_dataset, ADAM(), cb = evalcb)

p1 = jim(test_set_image[:,:,:,2])
p2 = jim(test_set_label[:,:,:,2])
p3 = jim(model_unet(test_set_image[:,:,:,2,:]).>0.5)

plot(p1,p2,p3)

#for i in size(test_set_image,4)

#end
