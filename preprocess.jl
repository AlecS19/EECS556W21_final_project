#preprocess
#Gathering Figures
## Test with color
include("DGMM.jl")
using Images: load, channelview
include("performance.jl")
using MIRT: jim
using Plots
using ImageFiltering
using Statistics
using ImageView

picnum = "61060"
filepath3 = "test_images/"* picnum *".jpg"
test_color = channelview(load(filepath3))
c,m,n = size(test_color)
gray = Gray.(load(filepath3))
grayFiltered = mapwindow( median, gray, (5,5))
filtered =  [mapwindow( median, test_color[1,:,:], (3,3) ), mapwindow( median, test_color[2,:,:], (7,7) ), mapwindow( median, test_color[3,:,:], (3,3) )]

jim(filtered[1]')
jim(filtered[2]')
jim(test_color[2,:,:]')
jim(filtered[3]')
