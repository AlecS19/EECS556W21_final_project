using Images: load, channelview
include("performance.jl")
using MIRT: jim
using Plots
#Testing DGMM


curr_dir = pwd()
filename1 = "figure4b.png"
filename2 = "figure4a.png"

filepath1 = "test_images/" * filename1
filepath2 = "test_images/" * filename2
correct = convert(Array{Float64}, load(filepath2)) .* 255

test = convert(Array{Float64}, load(filepath1)) .* 255 #right now the data type is not capable of handling the data when it is 0 -1
m, n = size(test)
test_flat = Array{Float64,2}(reshape(test, 1, :)')

include("DGMM.jl")
A = DGMM(test_flat, 4, (m, n))
E = log_lik(A, test_flat)
gray_history = train(A, test_flat, 4)
p1 = plot(gray_history)
display(p1)
D = post_prob(A)

ass = [argmax(D[i, :]) for i = 1:size(D, 1)]

centers = A.μ

val = [0, 80, 170, 255]
t = sortperm(centers[:])

segmented = ass
for i = 1:4
    segmented[segmented.==i] .= val[findall(x -> x == i, t)]
end
segmented = reshape(segmented, (128, 128))
img_seg = jim(segmented')

print(MCR(segmented, correct))
display(img_seg)

## Test with color
picnum = "61060"
filepath3 = "test_images/"* picnum *".jpg"
test_color = channelview(load(filepath3))
c, m, n = size(test_color)
test_color_flat = Array{Float64,2}([vec(test_color[1, :, :]) vec(test_color[2, :, :]) vec(test_color[3,:,:,])] .* 255,)
numsegs= 3
color_dgmm = DGMM(test_color_flat, numsegs, (m, n))

color_history = train(color_dgmm, test_color_flat, numsegs)
color_prob = post_prob(color_dgmm)
pcolor = plot(color_history)
ass=[argmax(color_prob[i,:]) for i=1:size(test_color_flat,1)]
segmented_gmm = reshape(ass, ( m,n ) ) .*(255/3)
jim(segmented_gmm')


using MAT
filepath4 = "ground_truth/" *picnum* ".mat"
file = matopen(filepath4)
correct = read(file, "groundTruth") # note that this does NOT introduce a variable ``varname`` into scope
close(file)
print( PR_fast(segmented_gmm,correct) )

p4 = jim(segmented_gmm')
display(p4)
display(pcolor)
