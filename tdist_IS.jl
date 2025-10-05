# This script takes one command-line argument, which is the dataset index.
data_idx = parse(Int64, ARGS[1]);

# E -> 0, 2A -> 0
# data_idx = 12;

# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));

OUTDIR = joinpath(@__DIR__, "output", "data$(data_idx)") # output directory

# Fetch packages.
using Combinatorics, Distributions, LinearAlgebra, LogExpFunctions, OrdinaryDiffEq, PDMats, PEtab, Random
using JLD2, ProgressMeter, Suppressor

n_extra = length(rxs_extra);
n_models = length(models);

# Integer combiantions
combs = [Int64[], combinations(collect(1:n_extra))...];
rx_boolmat = [Int(rx ∈ comb) for comb in combs, rx in 1:n_extra]; # 2^R by R
nparams = length.(combs) .+ 4;

@load "params.jld2" param_sets;
@load "data.jld2" all_data;

data = all_data[data_idx];
petab_models = [create_petab_model(model, data, u0) for model in models];
petab_probs = [PEtabODEProblem(pmodel; odesolver=ODESolver(Rodas5P(), verbose=false)) for pmodel in petab_models];

@load "$OUTDIR/MLE.jld2" model_fits
@load "$OUTDIR/MLE_hess.jld2" MLE_hessians

# fmins = getproperty.(model_fits, :fmin);
# logdets = -logdet.(MLE_hessians);
# LAs = -fmins .+ 0.5logdets + 0.5nparams*log(2π);

function laplace_IS(f, MAP, hess, n_samples; df=5, k=2, progress=false, time_when=[n_samples])
    T0 = time_ns()

    Σ = k * inv(PDMat(hermitianpart!(hess)))
    proposal = MvTDist(df, MAP, Σ)
    samples = rand(proposal, n_samples)

    time_when = sort(time_when, rev=true)
    times = Float64[]
    logws = Float64[]

    if progress p = Progress(n_samples; dt=1) end

    for (i, x) in enumerate(eachcol(samples))      
        push!(logws, -f(x) - logpdf(proposal, x))
        if (length(time_when) > 0) && (i == time_when[end])
            push!(times, (time_ns()-T0)/1e9)
            pop!(time_when)
        end
        progress && next!(p)
    end
    return logws, times
end

n_samples = 10^6;
time_when = Int64.([1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6])
shuffle_perm = randperm(length(petab_probs))
shuffled_IS_results = @showprogress [
    laplace_IS(
        petab_probs[model_idx].nllh,
        collect(model_fits[model_idx].xmin),
        MLE_hessians[model_idx],
        n_samples; time_when
    ) for model_idx in shuffle_perm
];

IS_results = shuffled_IS_results[invperm(shuffle_perm)];
# @suppress_err 
@save "$OUTDIR/IS.jld2" IS_results

# IS_logmean = logsumexp(logws) - log(n_samples)
# IS_logmeansq = logsumexp(2 .* logws) - log(n_samples)
# (IS_logmean, IS_logmeansq)

# IS_logmean = logsumexp(logws) - log(n_samples)
# IS_logmeansq = logsumexp(2 .* logws) - log(n_samples)
# IS_varlog = (exp(IS_logmeansq - 2IS_logmean) - 1) / n_samples
# sqrt(IS_varlog)
# ess = n_samples * exp(2IS_logmean - IS_logmeansq)

# LAs[model_idx]
