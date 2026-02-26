using LinearAlgebra, LinearAlgebra.LAPACK, LogExpFunctions
using Distributions, Random, StatsBase
using Distances

include(joinpath(@__DIR__, "stats_helpers.jl"));


"""
Represents a Gaussian mixture model with K components.
- `K::Int`: Number of mixture components
- `d::Int`: Dimensionality of the data
- `weights::Vector{Float64}`: Mixing coefficients
- `means::Vector{Vector{Float64}}`: Component means
- `chols::Vector{Cholesky{Float64, Matrix{Float64}}}`: Cholesky objects for component precisions
"""
struct GaussianMixture
    K::Int
    d::Int
    weights::Vector{Float64}
    means::Vector{Vector{Float64}}
    chols::Vector{Cholesky{Float64, Matrix{Float64}}}
end


"""Initialize a Gaussian mixture with K components from data X.
- `X::Matrix{Float64}`: Data matrix (d×N) where d is dimensionality and N is number of samples
- `K::Int`: Number of mixture components
- `method::Symbol`: Initialization method (:random or :kmeans_pp)
"""
function initialize_gm(X::Matrix{Float64}, K::Int; method::Symbol=:kmeans_pp)
    d, N = size(X)
    
    # Initialize means
    if method == :kmeans_pp
        means = kmeans_pp_init(X, K)
    elseif method == :random
        idxs = randperm(N)[1:K]
        means = [X[:, i] for i in idxs]
    else
        throw(ArgumentError("method must be :random or :kmeans_pp"))
    end
    
    # Initialize with spherical covariances
    overall_var = var(X; dims=2) |> vec
    precisions = [diagm(1 ./ overall_var) for _ in 1:K] # NB: Inflate precisions by d
    
    # Uniform weights
    weights = fill(1.0 / K, K)
    
    return GaussianMixture(K, d, weights, means, cholesky.(precisions))
end


"""
Add Gaussian components to an existing mixture with randomly sampled means.
"""
function augment_gm(gm::GaussianMixture, X::Matrix{Float64}, K_add::Int)
    d, N = size(X)
    idxs = randperm(N)[1:K_add]
    new_means = [X[:, i] for i in idxs]

    # Initialize with spherical covariances
    overall_var = var(X; dims=2) |> vec
    new_precisions = [diagm(1 ./ overall_var) for _ in 1:K_add] # NB: Inflate precisions by d

    # New weights
    new_K = gm.K + K_add
    old_prop = gm.K / new_K
    new_weights = [gm.weights .* old_prop; fill(1/new_K, K_add)]

    return GaussianMixture(new_K, d, new_weights, [gm.means; new_means], [gm.chols; cholesky.(new_precisions)])
end


function reset_gm!(gm::GaussianMixture, X::Matrix{Float64}, kvec::AbstractVector{Int})
    d, N = size(X)
    K_reset = length(kvec)

    # Randomly select means from X
    idxs = randperm(N)[1:K_reset]
    for (i, k) in zip(idxs, kvec)
        gm.means[k] .= X[:, i]
    end

    # Reset precision Cholesky with spherical covariances
    overall_var = var(X; dims=2) |> vec
    new_prec_chol = cholesky(diagm(d ./ overall_var))
    for k in kvec
        gm.chols[k].factors .= new_prec_chol.factors
    end

    # New weights
    gm.weights .*= 1 - K_reset / gm.K
    for k in kvec
        gm.weights[k] = 1 / K_reset
    end
end


"""
K-means++ initialization for cluster centers.
"""
function kmeans_pp_init(X::AbstractMatrix{Float64}, K::Int; 
                        init_means::AbstractVector{Vector{Float64}}=Vector{Float64}[]
)
    d, N = size(X)
    means = Vector{Float64}[]
    σ²_vec = vec(var(X; dims=2))
    dist_func = WeightedSqEuclidean(1 ./ σ²_vec)   
    
    # Remaining centers chosen with probability proportional to distance squared
    dists = fill(Inf, N)
    if !isempty(init_means)
        for init_mean in init_means
            push!(means, copy(init_mean))
            for i in 1:N
                dists[i] = min(dists[i], dist_func(view(X, :, i), init_mean))
            end
        end
    else 
        # First center chosen uniformly at random, distances are updated at the start of for loop below
        push!(means, X[:, rand(1:N)])
    end

    for k in (length(means)+1):K
        # Update dists
        for i in 1:N
            dists[i] = min(dists[i], dist_func(view(X, :, i), means[k-1]))
        end
        
        # Sample next center
        sample_weights = StatsBase.weights(dists ./ sum(dists))
        idx = sample(1:N, sample_weights)
        push!(means, X[:, idx])
    end
    
    return means
end


"""
Compute log PDF of multivariate normal in-place.
"""
function log_mvn_pdf!(result::AbstractVector{Float64}, X::AbstractMatrix{Float64}, 
                      μ::Vector{Float64}, chol::Cholesky{Float64, Matrix{Float64}}, 
                      diff_tmp::Vector{Float64}; quadform_only::Bool=false)
    d, N = size(X)
    log_det = logdet(chol)
    const_term = quadform_only ? 0. : 0.5 * (-d * log(2π) + log_det)
    
    for i in 1:N
        @inbounds for j in 1:d
            diff_tmp[j] = X[j, i] - μ[j]
        end
        lmul!(chol.U, diff_tmp)   
        mahal_sq = dot(diff_tmp, diff_tmp)
        result[i] = const_term - 0.5 * mahal_sq
    end
end


"""
E-step: Compute responsibilities (posterior probabilities) in-place and returns log likelihood.
"""
function e_step!(responsibilities::Matrix{Float64}, gm::GaussianMixture, 
                 X::AbstractMatrix{Float64}, log_probs::Matrix{Float64}, 
                 diff_tmp::Vector{Float64}, xweights::AbstractVector{Float64}, 
                 is_fixed::AbstractVector{Bool}, fixed_log_probs::Matrix{Float64})
    K, N = size(responsibilities)
    ll = 0.
    
    # Compute log probabilities for each component
    for k in 1:K
        if is_fixed[k]
            copyto!(view(log_probs, k, :), view(fixed_log_probs, k, :))
        else
            log_mvn_pdf!(view(log_probs, k, :), X, gm.means[k], gm.chols[k], diff_tmp)     
        end   
        lwk = log(gm.weights[k])
        @inbounds for i in 1:N
            log_probs[k, i] += lwk
        end
    end
    
    # Compute reponsibilities
    @inbounds for i in 1:N
        col = view(log_probs, :, i)
        lse = logsumexp(col)
        ll += xweights[i] == 0. ? 0. : lse * xweights[i]
        for k in 1:K
            responsibilities[k, i] = exp(col[k] - lse) * xweights[i]
        end
    end

    return ll
end


"""
M-step: Update GM parameters based on responsibilities.
"""
function m_step!(gm::GaussianMixture, X::AbstractMatrix{Float64}, 
                 responsibilities::Matrix{Float64}, Nk::AbstractVector{Float64},
                 pen_cov::Matrix{Float64}, pen_weight::Float64,
                 cov_tmp::Matrix{Float64}, diff_tmp::Vector{Float64},
                 is_fixed::AbstractVector{Bool};
                 cov_reg::Float64=1e-8)
    d, N = size(X)
    K = gm.K
    
    # Compute effective number of points assigned to each component
    Nk_sum = 0
    # weight_fixed = 0.
    for k in 1:K
        # if is_fixed[k] 
            # weight_fixed += gm.weights[k]
            # continue
        # end
        Nk[k] = sum(view(responsibilities, k, :))
        Nk_sum += Nk[k]
    end
    
    # Update weights
    # fct = (1 - weight_fixed) / Nk_sum
    for k in 1:K
        # is_fixed[k] && continue
        # gm.weights[k] = fct * Nk[k]
        gm.weights[k] = Nk[k] / Nk_sum
    end
    
    # Update means
    for k in 1:K
        is_fixed[k] && continue
        Nk[k] == 0. && continue
        fill!(gm.means[k], 0.)
        for i in 1:N
            r = responsibilities[k, i]
            @inbounds for j in 1:d
                gm.means[k][j] += r * X[j, i]
            end
        end
        gm.means[k] ./= Nk[k]
    end
    
    # Update covariances
    for k in 1:K
        is_fixed[k] && continue
        copyto!(cov_tmp, pen_cov)
        cov_tmp .*= 2 * pen_weight
        for i in 1:N
            r = responsibilities[k, i]
            @inbounds for j in 1:d
                diff_tmp[j] = X[j, i] - gm.means[k][j]
            end
            @inbounds for j1 in 1:d
                @inbounds for j2 in 1:d
                    cov_tmp[j1, j2] += r * diff_tmp[j1] * diff_tmp[j2]
                end
            end
        end
        cov_tmp ./= Nk[k] + 2 * pen_weight
        
        # Add regularization to prevent singular matrices
        @inbounds for j in 1:d
            cov_tmp[j, j] += cov_reg
        end

        hermitianpart!(cov_tmp) # Make symmetric

        all(isfinite, cov_tmp) || @info k Nk[k] Nk sum(Nk)

        # Compute precision Cholesky in-place
        prec_tmp = gm.chols[k].factors
        _, info = LAPACK.potrf!('L', cov_tmp)
        if info != 0
            error("Cholesky factorization of covariance failed, info = $info, eigvals = $(eigvals(cov_tmp))")
        end
        copyto!(prec_tmp, I)
        ldiv!(LowerTriangular(cov_tmp), prec_tmp)
        ldiv!(LowerTriangular(cov_tmp)', prec_tmp)
        _, info = LAPACK.potrf!('U', prec_tmp)
        if info != 0
            error("Cholesky factorization of covariance failed, info = $info, eigvals = $(eigvals(prec_tmp))")
        end
    end
end


"""
Fit a Gaussian mixture model using the EM algorithm.

# Arguments
- `gm::GaussianMixture`: Initial GM (will be modified in-place)
- `X::AbstractMatrix{Float64}`: Data matrix (d×N)
- `max_iter::Int`: Maximum number of EM iterations
- `tol::Float64`: Convergence tolerance for log likelihood
- `verbose::Bool`: Print progress information

# Returns
- `Vector{Float64}`: Log-likelihood values at each iteration
"""
function fit_gm!(gm::GaussianMixture, X::AbstractMatrix{Float64}; 
                 max_iter::Int=1000, tol::Float64=1e-8, verbose::Bool=false,
                 is_fixed::AbstractVector{Bool}=fill(false, gm.K),
                 xweights::AbstractVector{Float64}=ones(size(X,2)))
    d, N = size(X)
    K = gm.K
    
    # Pre-allocate workspace
    responsibilities = zeros(K, N)
    log_probs = zeros(K, N)
    Nk = zeros(K)
    diff_tmp = zeros(d)    
    cov_tmp = zeros(d, d)

    global_cov = cov(X, weights(xweights), 2)
    pen_weight = 1/compute_ess(log.(xweights))
    pen_func(gm) = pen_weight * sum(
        begin
            # mul!(cov_tmp, chol.U, chol.U')
            # cov_tmp .*= global_cov
            # logdet(chol) - sum(cov_tmp)
            logdet(chol) - tr(global_cov * Matrix(chol))
        end for chol in gm.chols
    )

    fixed_log_probs = zeros(K, N)
    for k in 1:K
        log_mvn_pdf!(view(fixed_log_probs, k, :), X, gm.means[k], gm.chols[k], diff_tmp) 
    end
    
    log_liks = Float64[]
    
    for iter in 1:max_iter
        # E-step
        ll = e_step!(
            responsibilities, gm, X, log_probs, diff_tmp, 
            xweights, is_fixed, fixed_log_probs
        ) + pen_func(gm)
        push!(log_liks, ll)
        
        if verbose
            @info "Iteration $iter" ll sum(gm.weights) sort(gm.weights)
        end
        
        # Check convergence
        if iter > 1 && abs(ll - log_liks[end-1]) < (abs(ll)+1)*tol
            if verbose
                println("Converged after $iter iterations")
            end
            break
        end
        
        # M-step
        m_step!(
            gm, X, responsibilities, Nk, 
            global_cov, pen_weight,
            cov_tmp, diff_tmp, is_fixed
        )
    end
    
    return log_liks
end


function trim_gm(gm::GaussianMixture, thres::Float64=0.; is_fixed::AbstractVector{Bool}=fill(false, gm.K))
    to_keep = findall((gm.weights .>= thres) .|| is_fixed)
    return GaussianMixture(
        length(to_keep),
        gm.d,
        gm.weights[to_keep] ./ sum(gm.weights[to_keep]),
        gm.means[to_keep],
        gm.chols[to_keep]
    )
end


import Base.rand
import Distributions.logpdf

function randmvn!(v::AbstractVector{Float64}, μ::AbstractVector{Float64}, prec_chol::Cholesky{Float64, Matrix{Float64}})
    randn!(v)
    ldiv!(prec_chol.U, v)
    v .+= μ
    return v
end

function rand(gm::GaussianMixture, n::Int; stratified=false)
    ks = stratified ? stratified_sampling(gm.weights, n) : sample(1:gm.K, weights(gm.weights), n)
    return rand(gm, ks)
end

function rand(gm::GaussianMixture, ks::AbstractVector{Int})
    n = length(ks)
    samples = zeros(gm.d, n)
    for (i, k) in enumerate(ks)
        randmvn!(view(samples, :, i), gm.means[k], gm.chols[k])
    end
    return samples
end

function logpdf(gm::GaussianMixture, X::AbstractMatrix{Float64})
    d, N = size(X)
    diff_tmp = zeros(d)
    log_probs = zeros(gm.K, N)
    for k in 1:gm.K        
        if gm.weights[k] == 0.
            log_probs[k, :] .= -Inf
            continue
        end
        log_mvn_pdf!(view(log_probs, k, :), X, gm.means[k], gm.chols[k], diff_tmp)        
        lwk = log(gm.weights[k])
        @inbounds for i in 1:N
            log_probs[k, i] += lwk
        end
    end
    return vec(logsumexp(log_probs; dims=1))
end


# Cauchy-simplex optimisation

"""
- `f`: Function to minimise
- `w`: Current state
- `d`: Descent direction
- `f_w`: Current function value
- `grad_w`: Current gradient
- `cache`: Storage for new value
- `eta_max`: Max learning rate
- `c1`: Armijo condition parameter
- `c2`: Backtracking shrinkage factor
- `max_iter`: Maximum line search iterations
"""
function armijo_line_search(f, w, d, f_w, grad_w, cache, eta_max, p;
                            zero_tol::Float64=1e-8,
                            c1::Float64=1e-3,
                            c2::Float64=0.5,
                            max_iter::Int=40)
    count = 0;
    eta = eta_max;
    grad_w_d = dot(grad_w, d);
    # Armijo condition: f(w_new) <= f(w) + c1 * grad · (w_new - w)
    while count < max_iter
        baseline_f = f_w + c1 * eta * grad_w_d
        update_point!(cache, w, d, eta; zero_tol)
        f_new = f(cache, p)
        # Check Armijo condition
        if f_new <= baseline_f
            return eta
        end
        eta *= c2
        count += 1
    end

    # If max iterations reached and new point is worse, return 0
    if count == max_iter && f_w < f(cache, p)
        return 0.0
    end

    return eta
end

# Helper function to compute updated point
function update_point!(cache, w, d, step_size; zero_tol::Float64=1e-8,)
    s = 0.
    for i in eachindex(w)
        v = w[i] - step_size*d[i]
        cache[i] = v <= zero_tol ? 0. : v
        s += cache[i]
    end
    cache ./= s
end

function cauchy_simplex(f, grad!, w_init::AbstractVector{Float64}, p;
    max_iter::Int=1000,
    zero_tol::Float64=1e-8,
    c1::Float64=1e-3,
    c2::Float64=0.5,
    verbose::Bool=false)

    w = copy(w_init)

    f_prev = Inf
    f_hist = Vector{Float64}(undef, max_iter)
    converged = false
    
    wnew_cache = similar(w_init)
    grad_cache = similar(w_init)
    d_cache = similar(w_init)

    for iter in 1:max_iter
        grad!(grad_cache, w, p)
        f_w = f(w, p)
        f_hist[iter] = f_w    

        w_dot_grad = dot(w, grad_cache)
        @inbounds for i in eachindex(d_cache) # ∇f - (w · ∇f) * 1
            d_cache[i] = grad_cache[i] - w_dot_grad
        end

        n_pos = 0
        max_Π_grad = 0.
        for i in eachindex(w)
            if w[i] > zero_tol
                n_pos += 1
                max_Π_grad = max(max_Π_grad, d_cache[i])
            end
        end
        if n_pos == 0 && verbose
            println("Warning: All weights below tolerance at iteration $iter")
            # @info "not enough pos" f_w [w grad_cache d_cache]
        end     
        
        eta_max = 0.99 / max_Π_grad
        d_cache .*= w
        eta = armijo_line_search(f, w, d_cache, f_w, grad_cache, wnew_cache, eta_max, p; c1, c2, zero_tol)
        
        update_point!(wnew_cache, w, d_cache, eta; zero_tol)

        # if isnan(eta) || any(isnan, wnew_cache)
        #     @info "bad update" f_w eta max_Π_grad [w wnew_cache grad_cache d_cache]
        # end

        sqnorm_w = 0.
        for i in eachindex(w)
            diff = (w[i] - wnew_cache[i])
            sqnorm_w += diff*diff
        end

        if max_Π_grad <= 0. || eta == 0. || sqrt(sqnorm_w) <= zero_tol || abs(f_prev - f_w) / (1 + f_w) <= zero_tol
            converged = true
            if verbose
                @info "Converged at iter $iter" f_w max_Π_grad eta sqnorm_w
            end
            f_hist = f_hist[1:iter]
            break
        end        
        
        # Verbose output
        if verbose && (iter % 10 == 0 || iter == 1)
            @info "Iter $iter" f_w norm(grad_cache) norm(d_cache) eta log2(eta_max / eta) norm(w .- wnew_cache) max_Π_grad
        end

        copyto!(w, wnew_cache)
        f_prev = f_w
    end
    
    return w, f_hist, converged
end

