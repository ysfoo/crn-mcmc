using LinearAlgebra, LogExpFunctions
using Distributions, Random, StatsBase


# Effective sample size
function compute_ess(logws)
    return exp(2logsumexp(logws) - logsumexp(2 .* logws))
end


# Sample categorical distribution using {i/n}_{i=0}^{n-1} instead of drawing from Unif(0, 1)
function stratified_sampling(ws, n=length(ws))
    edges = cumsum(ws ./ sum(ws))
    edges[end] = 1.0
    u = (rand(n) .+ (0:(n-1))) ./ n
    idxs = searchsortedfirst.(Ref(edges), u)
    return idxs
end


# Bisection search for a monotonic function
function bisection_search(trg, func, lo, hi; tol=1e-6)
    is_increasing = func(hi) >= func(lo)
    left, right = lo, hi
    while right - left > tol
        mid = (left + right) / 2
        val = func(mid)
        if (val <= trg) == is_increasing
            left = mid
        else
            right = mid
        end
    end
    
    return right
end


# Returns indices corresponding to largest log weights that covers 1-α of the total
function select_top(logws, α; n_max=length(logws))
    lse = logsumexp(logws)
    ord = sortperm(logws, rev=true)
    tmp = -Inf
    for (upto, idx) in enumerate(ord)
        tmp = logaddexp(logws[idx], tmp)
        if tmp >= lse + log1p(-α)
            return ord[1:min(n_max,upto)]
        end
    end
    return ord[1:n_max]
end


# Return `top` largest entries of `v`
function sortview(v; top=5)
    top = min(top, length(v))
    return sort(collect(enumerate(v)), by=last, rev=true)[1:top]
end
