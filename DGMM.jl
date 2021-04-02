#File defining the DGMM and related functions

using LinearAlgebra: det
using Clustering: kmeans
using ImageFiltering: imfilter, centered
## DGMM Structure

mutable struct DGMM
    ## Parameters
    β::Float64
    n::UInt8 #number of clusters
    d::UInt8#dimensions

    ## Information
    μ::Array{Float64,2} #n x d array of centers
    Σ::Array{Float64,2} #n x d^2 array of covariances
    k::UInt16 #Iteration Number

    function DGMM(n::Int64, d::Int64)
        new(500,n,d,zeros(n,d),ones(n,d^2),1)
    end

    function DGMM(x::Array{Float64,2}, n::Int64)
        #initialize DGMM with kmeans
        dgmm = DGMM(n, size(x,2))
        init( dgmm, x )
        return dgmm
    end
end

## DGMM Functions

#Find the final parameters based on data
function train(dgmm::DGMM, data::Array{Float64,2}, n::Int64 )
    #E-step

    #M-step
end

## DGMM Helper Functions

#Initialize the parameters based on k-means
function init(dgmm::DGMM, data::Array{Float64,2})

    #Kmeans initialize
    #TODO: Add additional parameters for kmeans
    results = kmeans(data', dgmm.n)
    dgmm.μ = results.centers'

    #Covariance initialization
    #TODO: Determine how to do find covariance

end

#Gaussian Distribution of a Matrix
function gaus_dist(dgmm::DGMM, data::Array{Float64,2})
    dist = zeros(  size(data,1), dgmm.n, )

    global cov = reshape(dgmm.Σ[1,:], (dgmm.d,dgmm.d))
    global mean = dgmm.μ[1,:]

    gaus_dist_exp = (x) -> (x .- mean)' * cov^(-1) * (x .- mean) ./-2

    for i in 1:dgmm.n
        global cov = reshape(dgmm.Σ[i,:], (dgmm.d,dgmm.d))
        global mean = dgmm.μ[i,:]

        coeff =  (2*π)^(dgmm.d/2) * sqrt( abs( det( cov ) ) )
        dist[:,i] = exp.( mapslices(gaus_dist_exp, data, dims=2) )
    end

    return dist
end

#Calculate the posterior probability
function post_prob(dgmm::DGMM, dist::Array{Float64,2}, dim)

    p = π_nk(dgmm, dist , dim ) .* dist

    return  p ./ sum(p, dims=2)
end

#Calculate the log-liklihood
function log_lik(dgmm::DGMM, data::Array{Float64,2})

    dist = gaus_dist( dgmm, data )
    #TODO: Check to see if the formulation is correct. Not sure where znk came from
    liklihood = post_prob(dgmm, data) .*( log10.( π_nk(dgmm, dist, size(data)) ) .+ log10.(dist) )

    #TODO: Handle times when the number is NaN
    return sum(liklihood[isnan.(liklihood) .== 0]) #this is dumb
end

#Calculate contextual mixing portion
function π_nk(dgmm::DGMM, dist::Array{Float64,2}, dims)

    window = centered( ones(3,3) )
    global mixprop = zeros( size(dist) )

    #Cycle through the number of segments
    for k in 1:dgmm.n
        mixprop[:,k] = exp.( dgmm.β.*vec( imfilter( reshape( dist[:,k], dims ), window ) ) )
    end

    mixprop = mixprop ./ (sum( mixprop, dims=2) .+ 1e-12) #TODO: make this a parameter, prevent dividing by 0

    return mixprop
end
