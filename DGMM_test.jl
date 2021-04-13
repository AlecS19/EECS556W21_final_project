using Images: load
include("performance.jl")
using MIRT: jim
#Testing DGMM


curr_dir = pwd()
filename1 = "figure4b.png"
filename2 ="figure4a.png"

filepath1 = "test_images/" * filename1
filepath2 = "test_images/" * filename2
correct = convert(Array{Float64}, load(filepath2) ).*255

test = convert(Array{Float64}, load(filepath1) ).*255 #right now the data type is not capable of handling the data when it is 0 -1
test_flat = Array{Float64,2}( reshape(test,1,:)' )

include("DGMM.jl")
A  = DGMM(test_flat, 4)
B = gaus_dist(A, test_flat)
C = π_nk(A,B, size(test_flat))
D = post_prob(A, B, size(test_flat))
E = log_lik(A, test_flat)
history = train(A,test_flat, 4)


B = gaus_dist(A, test_flat)
C = π_nk(A,B, size(test_flat))
D = post_prob(A, B, size(test_flat))


ass=[argmax(D[i,:]) for i=1:size(D,1)]

centers = A.μ

val = [0,80,170,255]
t = sortperm(centers[:])

segmented = ass
for i in 1:4
    segmented[segmented .== i] .= val[ findall(x->x==i,t) ]
end
segmented = reshape( segmented, (128,128) )
jim(segmented)

print(MCR(segmented,correct))
