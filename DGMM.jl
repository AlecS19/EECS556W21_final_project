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
    dims #dimension of an image

    ## Information
    μ::Array{Float64,2} #n x d array of centers
    Σ::Array{Float64,2} #n x d^2 array of covariances
    k::UInt16 #Iteration Number

    ## Intermediates
    dist::Array{Float64,2} #The gaussian distribution of the data at the current iteration
    mixprop::Array{Float64,2} #The mixture proportion of the current iteration
    post_p::Array{Float64,2} #The posterior probability of the current iteration

    function DGMM(n::Int64, d::Int64, dims)
        i = zeros(n,d)
        new(500,n,d, dims, #Paramters
        zeros(n,d),ones(n,d^2),1,#Information
        i,i,i #Intermediates
        )

    end

    function DGMM(x::Array{Float64,2}, n::Int64, dims)
        #initialize DGMM with kmeans
        dgmm = DGMM(n, size(x,2),dims)
        init( dgmm, x )

        i = ones(size(x,1),n)./n
        dgmm.dist = i
        dgmm.mixprop = i
        dgmm.post_p = i
        return dgmm
    end
end

## DGMM Functions

#Find the final parameters based on data
function train( dgmm::DGMM, data::Array{Float64,2}, n::Int64, nIter::Int64=80)
    global history = []
    append!(history, log_lik(dgmm, data))

    for i in 1:nIter
        dgmm.k = UInt8(i)
        println("Iteration $i")

        ####### E-step #########

        #Calculate Gaussian Distribution
        gaus_dist(dgmm,data)
        println("Gaussain $(dgmm.dist[1:5,:])")

        #Calculate mixing proportion
        π_nk(dgmm)
        println("Pi $(dgmm.mixprop[1:5,:])")

        #Calculate Posterior probability
        post_prob(dgmm)
        println("Posterior prob $(dgmm.post_p[1:5,:])")

        ######## M-step ########

        for k in 1:dgmm.n
            sum_post = sum(dgmm.post_p[:,k])

            #Update mean
            dgmm.μ[k,:] = sum(dgmm.post_p[:,k].*data, dims=1) ./ (sum_post )

            #Update covariance

            zero_mean_data = data .- Array{Float64,2}( dgmm.μ[k,:]' )

            #dgmm.Σ[k,:] =  sum( post_p[:,k] * reshape( (zero_mean_data'*zero_mean_data), (1,dgmm.d^2) ) , dims=1) ./ sum_post
            if dgmm.d == 1
                dgmm.Σ[k,:] = sum( dgmm.post_p[:,k] .* mapslices(X ->  X'*X, zero_mean_data, dims=2  ), dims=1) ./sum_post
            else
                dgmm.Σ[k,:] = sum( dgmm.post_p[:,k] .* mapslices(X ->vec( X*X'), zero_mean_data, dims=2  ), dims=1) ./sum_post
            end
        end

        #Update Beta
        gaus_dist(dgmm,data) #For some reason if you do not update this beta turns into NAN
        π_nk(dgmm)
        ϕ  = 10e-6 #Can be adjusted bu this is what the paper uses
        dgmm.β = dgmm.β - ϕ*partial_beta(dgmm)
        println("Beta $(dgmm.β)")

        #log liklihood
        append!( history, log_lik(dgmm, data) )
        println("Convergence $(abs.(history[i+1]))")

        #Check convergence
        #if abs.(history[i+1] - history[i]) / abs(history[i+1]) < 1e-5
            #break
        #end
    end
    return history
end

## DGMM Helper Functions

#Initialize the parameters based on k-means
function init(dgmm::DGMM, data::Array{Float64,2})

    #Kmeans initialize
    results = kmeans(data', dgmm.n)
    dgmm.μ = results.centers'

    #Covariance initialization

    for i in 1:dgmm.n
        selector = results.assignments .== i
        cluster = mapslices( x -> x[selector], data, dims = 1 )

        #zero mean
        cluster = cluster .- dgmm.μ[i,:]'
        dgmm.Σ[i,:] = reshape( cluster' * cluster , (dgmm.d^2,1)) ./ size(cluster,1)
        #dgmm.Σ[i,:] = reshape( data' * data , (dgmm.d^2,1)) ./ size(data,1)
    end

end

#Gaussian Distribution of a Matrix
function gaus_dist(dgmm::DGMM, data::Array{Float64,2})
    gaus_dist_exp = (x,cov, mean) -> (x .- mean)' * cov^(-1) * (x .- mean) ./(-2)

    for i in 1:dgmm.n
        cov = reshape( dgmm.Σ[i,:], (dgmm.d,dgmm.d) )
        mean = dgmm.μ[i,:]
        coeff =  (2*π)^(dgmm.d/2) * sqrt( abs( det( cov ) ) )
        dgmm.dist[:,i] = exp.( mapslices(x -> gaus_dist_exp(x,cov,mean), data, dims=2) ) ./coeff
        #dist[:,i] = exp.( (data .- mean) *( cov^(-1) ./(-2) )* (data .- mean)' ) ./ coeff
    end

    return dgmm.dist
end

#Calculate the posterior probability
function post_prob(dgmm::DGMM)
    dgmm.post_p = dgmm.mixprop .* dgmm.dist
    dgmm.post_p = dgmm.post_p ./ (sum(dgmm.post_p, dims=2))
    return  dgmm.post_p
end

#Calculate teh partial beta for beta update term
function partial_beta(dgmm::DGMM)
        #create the f function
        f =  dgmm.mixprop .* dgmm.dist
        println("f $(f[1:5,:])")

        window = centered( ones(3,3) ) #neighborhood

        #Initialize some of the vectors
        global f_neigh = zeros( size(dgmm.dist) )
        global denom = zeros( (size(dgmm.dist,1),1) )
        global numer = denom

        for k in 1:dgmm.n
            global f_neigh[:,k] = vec( imfilter( reshape(f[:,k], dgmm.dims), window ) )
            f_exp = exp.(dgmm.β.*f_neigh[:,k])

            #cumulate the denominator
            global denom .+= f_exp
            f_exp .*= f_neigh[:,k]

            #cumulate the numerator
            global numer .+= f_exp
        end

        total = zeros( (size(dgmm.dist,1),1) )
        for k in 1:dgmm.n
            total .+= dgmm.post_p[:,k] .* (f_neigh[:,k] .- (numer./denom))
        end

        return sum(total)
end

#Calculate the log-liklihood
function log_lik(dgmm::DGMM, data::Array{Float64,2})

    liklihood = dgmm.post_p .*( log.( dgmm.mixprop ) .+ log.(dgmm.dist) )
    liklihood = sum( log.( sum( dgmm.mixprop .* dgmm.dist ,dims=2) ) )

    #return sum(liklihood[isnan.(liklihood) .== 0]) #this is dumb

    return sum(liklihood)
end

#Calculate contextual mixing portion
function π_nk(dgmm::DGMM)

    window = centered( ones(3,3) )

    #Cycle through the number of segments
    for k in 1:dgmm.n
        dgmm.mixprop[:,k] =  exp.(dgmm.β.*vec( imfilter( reshape( dgmm.dist[:,k], dgmm.dims ), window ) ))
    end
    dgmm.mixprop = dgmm.mixprop ./ sum( dgmm.mixprop ,dims=2)

    return dgmm.mixprop
end
