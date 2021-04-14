using Images
using FileIO: save
using NIfTI
using GZip
using MIRTjim:jim
using Statistics:mean

Image_path = "Brain_dataset\\"

filenames = readdir(Image_path)


test = niread(Image_path * filenames[7])
test2 = niread(Image_path * filenames[8])
#test2 = niread(Label_path * filenames[1])

jim(test)
jim(test2)

images = zeros(256,256,1,39)
annotations = zeros(256,256,1,7)
labels = zeros(256,256,1,39)


for i in 31:39
   images[:,:,1,i] = niread(Image_path * "2D_qubiq_fetal_" * lpad(i,2,"0") * "_image.nii")
   if i == 36
      labels[:,:,1,i] = niread(Image_path * "2D_qubiq_fetal_" * lpad(i,2,"0") * "_annotation_01.nii")
   else
      for j in 1:7
         annotations[:,:,1,j] = niread(Image_path * "2D_qubiq_fetal_" * lpad(i,2,"0") * "_annotation_0" * string(j) * ".nii")
      end
      labels[:,:,1,i] = mean(annotations.>0.1,dims=4).>0.5
   end
end

train_set_image = convert(Array{Float32,4}, images[:,:,:,1:32])
train_set_label = convert(Array{Float32,4}, labels[:,:,:,1:32])

test_set_image = convert(Array{Float32,4}, images[:,:,:,33:39])
test_set_label = convert(Array{Float32,4}, labels[:,:,:,33:39])


save(Image_path * "Train_Set_Images.jld2", "train_set_image", train_set_image)
save(Image_path * "Train_Set_Labels.jld2", "train_set_label", train_set_label)
save(Image_path * "Test_Set_Images.jld2", "test_set_image", test_set_image)
save(Image_path * "Test_Set_Label.jld2", "test_set_label", test_set_label)
