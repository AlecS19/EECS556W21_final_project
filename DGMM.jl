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
function train(dgmm::DGMM, data::Array{Float64,2}, n::Int64, nIter::Int64=200 )
    global history = []
    append!(history, 0)
    dim = size(data)

    for i in 1:nIter
        ####### E-step #########

        #Calculate Gaussian Distribution
        dist = gaus_dist(dgmm, data)

        #Calculate mixing proportion
        mixprop = π_nk(dgmm, dist, dim)

        #Calculate Posterior probability
        post_p = post_prob(dgmm, dist, dim)
        print(post_p)

        ######## M-step ########

        for k in dgmm.n
            sum_post = sum(post_p[:,k])
            #Update mean
            print( sum(post_p[:,k].*data) ./ sum_post )
            dgmm.μ[k,:] = sum(post_p[:,k].*data) ./ sum_post

            #Update covariance
            zero_mean_data = data[:,k] .- dgmm.μ[k,:]
            dgmm.Σ[k,:] = vec( post_p[:,k].*( ( zero_mean_data )' * ( zero_mean_data ) ) ./ sum_post )
        end

        #Update Beta
        #TODO: implement gradient descent to update Beta
        dist = gaus_dist(dgmm, dist)

        ϕ  = 10e-6 #Can be adjusted bu this is what the paper uses
        dgmm.β = dgmm.β - ϕ*partial_beta(dgmm, dist, dim)

        #log liklihood
        append!( history, log_lik(dgmm, data) )

        #Check convergence
        if abs.(history[end] - history[end -1] / history[end]) <= 10e-5
            break
        end
    end
    return history
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

function partial_beta(dgmm::DGMM, dist::Array{Float64,2}, dim)
        f =  π_nk(dgmm, dist, dim) .* dist
        f_neigh = []
        window = centered( ones(3,3) )

        global denom = 0

        for k in 1:dgmm.n
            append!(f_neigh, vec( imfilter( reshape(f[:,k], dim), window ) ) )
            f_exp = exp.(dgmm.β.*f_neigh[k])

            denom += f_exp

            f_exp .*= f_neigh[k]
        end

        f_exp ./= denom

        total = zeros( size(dist,1) )
        for k in 1:dgmm.n
            total .+= post_prob(dgmm, dist, dim)[:,k] .* (f_neigh[k] .+ f_exp)
        end

        return sum(total)
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
