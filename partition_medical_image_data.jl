using Images
using FileIO: save


Image_path = "medical_dataset\\Images"
Label_path = "medical_dataset\\Labels"

filenames = readdir(Image_path)

images = zeros(512,512,1,100)
labels = zeros(512,512,1,100)

for i in 1:100
   images[:,:,1,i] = load("medical_dataset\\Images\\" * filenames[i])'
   labels[:,:,1,i] = abs.(load("medical_dataset\\Label_PNG\\" * filenames[i]))'.>0.001
end

train_set_image = convert(Array{Float32,4}, images[:,:,:,21:100])
train_set_label = convert(Array{Float32,4}, labels[:,:,:,21:100])

test_set_image = convert(Array{Float32,4}, images[:,:,:,1:20])
test_set_label = convert(Array{Float32,4}, labels[:,:,:,1:20])


save("medical_dataset\\Train_Set_Images.jld2", "train_set_image", train_set_image)
save("medical_dataset\\Train_Set_Labels.jld2", "train_set_label", train_set_label)
save("medical_dataset\\Test_Set_Images.jld2", "test_set_image", test_set_image)
save("medical_dataset\\Test_Set_Label.jld2", "test_set_label", test_set_label)
