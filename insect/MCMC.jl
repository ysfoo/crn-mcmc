# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
dir_idx = parse(Int64, ARGS[1])
genmodel_idx = feasible_idxs[dir_idx]

OUTDIR = mkpath(joinpath(@__DIR__, "output", "data$(dir_idx)")) # output directory
INFDIR = joinpath(@__DIR__, "scratch_output/data$(dir_idx)");

# Fetch packages.
using PEtab, OrdinaryDiffEq
using JLD2, ProgressMeter, Random, StableRNGs, Suppressor
using AdvancedHMC, Bijectors, LinearAlgebra, LogDensityProblems, LogDensityProblemsAD, MCMCChains, Turing

# Load ground truth, data, MAP results.
@load joinpath(@__DIR__, "data.jld2") all_data;
@load "$OUTDIR/MAP.jld2" model_fits;

data = all_data[genmodel_idx];

@showprogress for model_idx in 1:n_models 
    mcmc_fname = joinpath(INFDIR, "MCMC_model$(model_idx).jld2")
    isfile(mcmc_fname) && continue

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
    seed = model_idx
    nadapts = 2000
    while true
        rng = StableRNG(seed)
        chn = sample(
            rng, turing_model(target), Turing.NUTS(0.9), 10000; 
            initial_params=InitFromParams((θ=init_params,)), 
            nadapts=nadapts, save_state=false, progress=false
        );
        acc_rate = round(mean(vec(chn[:acceptance_rate])); digits=4)
        ess_df = ess(chn)
        min_ess = round(Int, minimum(ess_df.nt.ess))
        duration = round(MCMCChains.compute_duration(chn)/60; digits=2)
        println("Model $(model_idx), acc rate = $acc_rate, min_ess = $min_ess, time = $duration min")    
        flush(stdout)
        flush(stderr)

        if min_ess < 500
            seed += 100
            nadapts += 1000
            continue
        end

        nMCMC = size(chn)[1]
        @suppress_err @save mcmc_fname chn ess_df nMCMC;
        break
    end
end
