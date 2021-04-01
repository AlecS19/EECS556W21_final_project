#File defining the DGMM and related functions


# DGMM Structure

mutable struct DGMM
    ## Parameters
    π_nk::Float64
    β::Float64
    n::UInt8 #number of clusters
    d::UInt8#dimensions

    ## Information
    μ::Array{Float64,2} #n x d array of centers
    Σ::Array{Float64,2} #n x d^2 array of covariances
    k::UInt16 #Iteration Number

    function DGMM()
        new(1,500,1,1,zeros(1,1),zeros(1,1),1)
    end

    function DGMM(x::Array{Float64,2}, n::Int64)
        #initialize DGMM with kmeans
    end
end

#Initialize the parameters based on k-means
function init(dgmm::DGMM)
end

#Gaussian Distribution of a Matrix
function gaus_dist()
end

#Calculate the posterior probability
function post_prob()
end

#Calculate the log-liklihood
function log_lik()
end

#Find the final parameters based on data
function train(dgmm::DGMM, data::Array{Float64,2}, n::Int64 )
    dgmm.n = n
    print("Testing $(dgmm.n)")
end
