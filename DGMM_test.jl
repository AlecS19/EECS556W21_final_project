using Images: load
#Testing DGMM
include("DGMM.jl")

curr_dir = pwd()
filename1 = "figure4b.png"
filename2 ="figure4a.png"

filepath1 = curr_dir * "/EECS556W21_final_project/test_images/" * filename1
test = convert(Array{Float64}, load(filepath1) ).*255 #right now the data type is not capable of handling the data when it is 0 -1
test_flat = Array{Float64,2}( reshape(test,1,:)' )

A  = DGMM(test_flat, 4)
B = gaus_dist(A, test_flat)
C = π_nk(A,B, size(test_flat))
D = post_prob(A, B, size(test_flat))
E = log_lik(A, test_flat)
history = train(A,test_flat, 4)
