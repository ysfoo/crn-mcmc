# Run `setup.jl` and `../gaussian_mixtures.jl`.
include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../gaussian_mixtures.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
# dir_idx = parse(Int64, ARGS[1])
dir_idx = 2
genmodel_idx = feasible_idxs[dir_idx]

INFDIR = joinpath("/scratch/punim0638/ysfoo/crn-mcmc/insect/output/data$(dir_idx)") # inference result directory

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, PEtab, Random
using JLD2, ProgressMeter
using AdvancedHMC, Bijectors, BridgeSampling, LogDensityProblems, LogDensityProblemsAD, MCMCChains, PSIS, Turing

@load joinpath(@__DIR__, "data.jld2") all_data;
data = all_data[genmodel_idx];

function bridge_sampling(target, d, chn)
    trace = chn.value[:,1:d,1].data;
    samples = Matrix(trace');
    d = size(samples, 1)
    logp_samples = vec(chn[:logjoint]);
    logp_func(x) = begin
        res = target.logtarget(x)
        isnan(res) ? -Inf : res
    end

    smp1 = samples[:,1:2:end];
    smp2 = samples[:,2:2:end];
    logp_smp2 = logp_samples[2:2:end];
    n_post = size(smp1, 2)
    n_fit = size(smp2, 2)
    n_prop = 1000000

    # Fit mixture        
    K = 50;
    gm_fits = [
        begin
            Random.seed!(seed)
            gm = initialize_gm(smp2, K; method=:random)
            log_liks = fit_gm!(gm, smp2; max_iter=200)
            (gm=gm, em_objs=log_liks)
        end for seed in 1:10
    ];
    best_fit = argmax((x)->last(x.em_objs), gm_fits);
    gm = best_fit.gm

    # Propose from mixture, then evaluate all densities
    mix_smp = rand(gm, n_prop);
    logp_smp1 = logp_samples[1:2:end];
    logmix_smp1 = logpdf(gm, smp1);
    logp_mix = logp_func.(eachcol(mix_smp));
    logmix_mix = logpdf(gm, mix_smp);

    logml_mix, i_mix = BridgeSampling.iterative_algorithm(logp_smp1 .- logmix_smp1, logp_mix .- logmix_mix, n_post, n_prop; tol=1e-8, maxiter=1000, use_ess=true)
    LML_mix = BridgeSampling.LogMarginalLikelihood(logml_mix, i_mix, logp_smp1, logmix_smp1, logp_mix, logmix_mix)

    return LML_mix
end


@showprogress for model_idx in 1:n_models
    println("Model $(model_idx)")
    fname = joinpath(INFDIR, "BS_model$(model_idx).jld2")
    flush(stdout); flush(stderr);

    d = nparams[model_idx]
    pmodel = create_petab_model(models[model_idx], data, u0)
    petab_prob = PEtabODEProblem(pmodel; odesolver=ODESolver(Rodas5P(), verbose=false))
    target = PEtabLogDensity(petab_prob)

    mcmc_fname = joinpath(INFDIR, "alt_MCMC_model$(model_idx).jld2");
    @nowarn_load mcmc_fname chn;
    trace = chn.value[:,1:d,1].data;

    Random.seed!(dir_idx*n_models + model_idx)
    timed_res = @timed bridge_sampling(target, d, chn)    
    @save fname timed_res
end