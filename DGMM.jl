#File defining the DGMM and related functions

using LinearAlgebra: det
using Clustering: kmeans
using ImageFiltering: imfilter, centered
## DGMM Structure

mutable struct DGMM
    ## Parameters
    β::Float64
    n::UInt8 #number of clusters
    d::UInt8 #dimensions

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
function train(dgmm::DGMM, data::Array{Float64,2}, n::Int64, nIter::Int64=30 )
    global history = []
    append!(history, 0)
    dim = size(data)

    for i in 1:nIter
        println("Iteration $i")
        ####### E-step #########

        #Calculate Gaussian Distribution
        dist = gaus_dist(dgmm, data)

        #Calculate mixing proportion
        mixprop = π_nk(dgmm, dist, dim)

        #Calculate Posterior probability
        post_p = post_prob(dgmm, dist, dim)

        ######## M-step ########

        for k in 1:dgmm.n
            sum_post = sum(post_p[:,k])

            #Update mean
            dgmm.μ[k,:] = sum(post_p[:,k].*data, dims=1) ./ (sum_post )

            #Update covariance
            zero_mean_data = data .- dgmm.μ[k,:]

            #dgmm.Σ[k,:] =  sum( post_p[:,k] * reshape( (zero_mean_data'*zero_mean_data), (1,dgmm.d^2) ) , dims=1) ./ sum_post

            dgmm.Σ[k,:] = sum( post_p[:,k] .* mapslices(X -> X'*X  , zero_mean_data, dims=2  ), dims=1) ./sum_post

        end

        #Update Beta
        #TODO: implement gradient descent to update Beta
        dist = gaus_dist(dgmm, data)

        ϕ  = 10e-4 #Can be adjusted bu this is what the paper uses
        dgmm.β = dgmm.β - ϕ*partial_beta(dgmm, dist, dim)

        #log liklihood
        append!( history, log_lik(dgmm, data) )

        #Check convergence
        if abs.(history[i+1] - history[i] / history[i+1]) <= 10e-5
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

    for i in 1:dgmm.n
        selector = results.assignments .== i

        cluster = mapslices( x -> x[selector], data, dims = 1 )

        #zero mean
        cluster = cluster .- dgmm.μ[i,:]'

        dgmm.Σ[i,:] = reshape( cluster' * cluster , (dgmm.d^2,1)) ./ size(data,1)
    end


end

#Gaussian Distribution of a Matrix
function gaus_dist(dgmm::DGMM, data::Array{Float64,2})
    dist = zeros(  size(data,1), dgmm.n, )
    gaus_dist_exp = (x,cov, mean) -> (x .- mean)' * cov^(-1) * (x .- mean) ./(-2)

    for i in 1:dgmm.n
        cov = reshape( dgmm.Σ[i,:], (dgmm.d,dgmm.d) )
        mean = dgmm.μ[i,:]
        coeff =  (2*π)^(dgmm.d/2) * sqrt( abs( det( cov ) ) )

        dist[:,i] = exp.( mapslices(x -> gaus_dist_exp(x,cov,mean), data, dims=2) ) ./coeff
        #dist[:,i] = exp.( (data .- mean) *( cov^(-1) ./(-2) )* (data .- mean)' ) ./ coeff
    end

    return dist
end

#Calculate the posterior probability
function post_prob(dgmm::DGMM, dist::Array{Float64,2}, dim)
    temp = π_nk(dgmm, dist , dim )
    p = temp .* dist
    return  p ./ (sum(p, dims=2))
end

#Calculate teh partial beta for beta update term
function partial_beta(dgmm::DGMM, dist::Array{Float64,2}, dim)
        #create the f function
        f =  π_nk(dgmm, dist, dim) .* dist
        window = centered( ones(3,3) ) #neighborhood

        #Initialize some of the vectors
        global f_neigh = zeros( size(dist) )
        global denom = zeros( (size(dist,1),1) )
        global numer = denom

        for k in 1:dgmm.n
            global f_neigh[:,k] = vec( imfilter( reshape(f[:,k], dim), window ) )
            f_exp = exp.(dgmm.β.*f_neigh[:,k])

            #cumulate the denominator
            global denom .+= f_exp
            f_exp .*= f_neigh[:,k]

            #cumulate the numerator
            global numer .+= f_exp
        end

        total = zeros( (size(dist,1),1) )
        for k in 1:dgmm.n
            total .+= post_prob(dgmm, dist, dim)[:,k] .* (f_neigh[:,k] .- (numer./denom))
        end

        return sum(total)
end

#Calculate the log-liklihood
function log_lik(dgmm::DGMM, data::Array{Float64,2})

    dist = gaus_dist( dgmm, data )
    #TODO: Check to see if the formulation is correct. Not sure where znk came from
    liklihood = post_prob(dgmm, dist, size(data)) .*( log.( π_nk(dgmm, dist, size(data)) ) .+ log.(dist) )

    #TODO: Handle times when the number is NaN
    #return sum(liklihood[isnan.(liklihood) .== 0]) #this is dumb
    return sum(liklihood)
end

#Calculate contextual mixing portion
function π_nk(dgmm::DGMM, dist::Array{Float64,2}, dims)

    window = centered( ones(3,3) )
    global mixprop = zeros( size(dist) )

    #Cycle through the number of segments

    for k in 1:dgmm.n
        mixprop[:,k] =  dgmm.β.*vec( imfilter( reshape( dist[:,k], dims ), window ) )
    end

    mixprop = exp.(mixprop .- log.( (sum( exp.(mixprop), dims=2) .+ 1e-8) )) #TODO: make this a parameter, prevent dividing by 0

    return mixprop
end
