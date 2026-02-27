include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../stats_helpers.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
dir_idx = parse(Int64, ARGS[1])
# dir_idx = 2
genmodel_idx = feasible_idxs[dir_idx]

INFDIR = joinpath("/scratch/punim0638/ysfoo/crn-mcmc/insect/output/data$(dir_idx)") # inference result directory

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, PEtab, Random
using JLD2, ProgressMeter
using Bijectors, LogDensityProblems, LogDensityProblemsAD
using PSIS

@load joinpath(@__DIR__, "data.jld2") all_data;
data = all_data[genmodel_idx];

OUTDIR = joinpath(@__DIR__, "output", "data$(dir_idx)") # output directory
@nowarn_load "$OUTDIR/MAP.jld2" model_fits;
@load "$OUTDIR/MAP_hess.jld2" MAP_hessians;

function laplace_IS(target, MAP, hess, n_samples; df=4)
    Σ = inv(PDMat(hermitianpart!(hess)))
    proposal = MvTDist(df, MAP, Σ)
    samples = rand(proposal, n_samples)

    logps = target.logtarget.(eachcol(samples))
    logqs = logpdf.(Ref(proposal), eachcol(samples))
    logws = logps .- logqs

    psis_res = psis(logws; normalize=false, warn=false)

    return (
        logps = logps,
        logqs = logqs,
        psis_logws = psis_res.log_weights,
        pareto_shape = psis_res.pareto_shape
    )
end

for model_idx in 1:n_models
    fname = joinpath(INFDIR, "laplace_IS_model$(model_idx).jld2")
    flush(stdout); flush(stderr);
    # isfile(fname) && continue

    d = nparams[model_idx];
    pmodel = create_petab_model(models[model_idx], data, u0);
    petab_prob = PEtabODEProblem(pmodel; odesolver=ODESolver(Rodas5P(), verbose=false));
    target = PEtabLogDensity(petab_prob);   
    # MAP0 = collect(model_fits[model_idx].xmin)
    # MAP1 = to_prior_scale(model_fits[model_idx].xmin, target)
    # MAP2 = target.inference_info.bijectors(MAP1)
    MAP = collect(model_fits[model_idx].xmin)
    hess = MAP_hessians[model_idx]
    timed_res = @timed laplace_IS(target, MAP, hess, 10^6)
    @save fname timed_res
    println("Model $(model_idx), $(timed_res.time/60) min")
end

# Test
# model_idx = 20;
# fname = joinpath(INFDIR, "laplace_IS_model$(model_idx).jld2");
# @load fname timed_res;
# timed_res.time
# N = length(timed_res.value.psis_logws);
# logsumexp(timed_res.value.psis_logws) - log(N)
# compute_ess(timed_res.value.psis_logws)
# timed_res.value.pareto_shape
