#Gathering Figures
## Test with color
include("DGMM.jl")
using Images: load, channelview
include("performance.jl")
using MIRT: jim
using Plots
using ImageFiltering
using Statistics

picnum = "227092"
filepath3 = "test_images/"* picnum *".jpg"
img = load(filepath3)
test_color = channelview(img)
c, m, n = size(test_color)
test_filtered = zeros(m*n, c)
test_gray = vec( mapwindow( median, Array{Float64,2}(Gray.(img)), (9,9)) )
for i in 1:c
    test_filtered[:,i] = vec( mapwindow( median, Array{Float64,2}(test_color[i,:,:]), (5,5) ) ).*255
end
test_color_flat = Array{Float64,2}([vec(test_color[1, :, :]) vec(test_color[2, :, :]) vec(test_color[3,:,:,]) test_gray].* 255)
numsegs = 3
color_dgmm = DGMM(test_filtered, numsegs, (m, n))

color_history = train(color_dgmm, test_filtered, numsegs)
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

jim(segmented_gmm')
display(pcolor)
