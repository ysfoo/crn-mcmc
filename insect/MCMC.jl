# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
dir_idx = parse(Int64, ARGS[1])
genmodel_idx = feasible_idxs[dir_idx]

OUTDIR = mkpath(joinpath(@__DIR__, "output", "data$(dir_idx)")) # output directory
MCMCDIR = joinpath("/scratch/punim0638/ysfoo/crn-mcmc/insect/output/data$(dir_idx)") # MCMC trace directory

# Fetch packages.
using PEtab, OrdinaryDiffEq
using JLD2, ProgressMeter, Random, StableRNGs, Suppressor
using AdvancedHMC, Bijectors, LinearAlgebra, LogDensityProblems, LogDensityProblemsAD, MCMCChains, Turing

# Load ground truth, data, MAP results.
@load joinpath(@__DIR__, "data.jld2") all_data;
@load "$OUTDIR/MAP.jld2" model_fits;

data = all_data[genmodel_idx];

@showprogress for model_idx in 1:n_models    
    pmodel = create_petab_model(models[model_idx], data, u0)
    petab_prob = PEtabODEProblem(pmodel; odesolver=ODESolver(Rodas5P(), verbose=false))
    target = PEtabLogDensity(petab_prob);
    init_params = target.inference_info.bijectors(to_prior_scale(model_fits[model_idx].xmin, target))
    
    @model function turing_model(target)
        θ ~ filldist(Turing.Flat(), target.dim)
        Turing.@addlogprob! target.logtarget(θ)
        return nothing
    end

    # run MCMC chain...
    mcmc_fname = joinpath(MCMCDIR, "gm_MCMC_model$(model_idx).jld2")
    if isfile(mcmc_fname)
        continue
    end
    rng = StableRNG(model_idx)
    chn = sample(
        rng, turing_model(target), Turing.NUTS(0.9; metricT=AdvancedHMC.UnitEuclideanMetric), 10000; 
        initial_params=InitFromParams((θ=init_params,)), 
        nadapts=1000, save_state=false, progress=false
    );
    acc_rate = round(mean(vec(chn[:acceptance_rate])); digits=4)
    ess_df = ess(chn)
    min_ess = round(Int, minimum(ess_df.nt.ess))
    println("Model $(model_idx), acc rate = $acc_rate, min_ess = $min_ess")    
    nMCMC = size(chn)[1]
    @suppress_err @save mcmc_fname chn ess_df nMCMC;
    flush(stdout)
    flush(stderr)
end
