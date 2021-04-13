using Images
using FileIO: save


Image_path = "Dataset\\Images"
Label_path = "Dataset\\Label_PNG"

filenames = readdir(Image_path)

images = zeros(512,512,1,100)
labels = zeros(512,512,1,100)

for i in 1:100
   images[:,:,1,i] = load("Dataset\\Images\\" * filenames[i])'
   labels[:,:,1,i] = abs.(load("Dataset\\Label_PNG\\" * filenames[i]))'.>0.001
end

train_set_image = convert(Array{Float32,4}, images[:,:,:,21:100])
train_set_label = convert(Array{Float32,4}, labels[:,:,:,21:100])

test_set_image = convert(Array{Float32,4}, images[:,:,:,1:20])
test_set_label = convert(Array{Float32,4}, labels[:,:,:,1:20])


save("Train_Set_Images.jld2", "train_set_image", train_set_image)
save("Train_Set_Labels.jld2", "train_set_label", train_set_label)
save("Test_Set_Images.jld2", "test_set_image", test_set_image)
save("Test_Set_Label.jld2", "test_set_label", test_set_label)
