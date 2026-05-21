# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
# dir_idx = 2
dir_idx = parse(Int64, ARGS[1])
genmodel_idx = feasible_idxs[dir_idx]

OUTDIR = mkpath(joinpath(@__DIR__, "output", "data$(dir_idx)")) # output directory
mkpath(OUTDIR)

# Fetch packages.
using PEtab, OrdinaryDiffEq
using JLD2, ProgressMeter, Random, PDMats, StableRNGs, Suppressor
using AdvancedHMC, Bijectors, LinearAlgebra, LogDensityProblems, LogDensityProblemsAD, MCMCChains, Turing

using ThreadPinning
isslurmjob() = get(ENV, "SLURM_JOBID", "") != ""
isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);

# Load ground truth, data, MAP results.
@load joinpath(@__DIR__, "data.jld2") all_data;
@load "$OUTDIR/MAP.jld2" model_fits;
@load "$OUTDIR/MAP_hess.jld2" MAP_hessians;

data = all_data[genmodel_idx];

function main(model_idx)
    mcmc_fname = joinpath(OUTDIR, "chains_model$(model_idx).jld2")
    isfile(mcmc_fname) && return false
    
    nadapts = 1000
    n_sample = 3000
    n_chains = 5

    pmodel = create_petab_model(models[model_idx], data, u0)
    petab_prob = PEtabODEProblem(pmodel; odesolver=ODESolver(Rodas5P(), verbose=false))
    target = PEtabLogDensity(petab_prob);
    MAP = model_fits[model_idx].xmin
    hess = MAP_hessians[model_idx]
    Σ = inv(PDMat(hermitianpart!(hess)))
    inits = rand(MvTDist(4, MAP, Σ), n_chains)
    init_params = [
        begin
            p = to_prior_scale(init_col, target) |> target.inference_info.bijectors
            InitFromParams((θ=p,))
        end for init_col in eachcol(inits)
    ]
    
    @model function turing_model(target)
        θ ~ filldist(Turing.Flat(), target.dim)
        Turing.@addlogprob! target.logtarget(θ)
        return nothing
    end

    # run MCMC chain...
    seed = model_idx   
    begin
        rng = StableRNG(seed)
        chn = sample(
            rng, turing_model(target), Turing.NUTS(0.9, metricT=AdvancedHMC.UnitEuclideanMetric), MCMCThreads(), n_sample, n_chains; 
            initial_params=init_params, 
            nadapts=nadapts, save_state=false, progress=false
        );
        acc_rates = collect(vec(mean(chn[:acceptance_rate]; dims=1)))
        step_sizes = collect(vec(chn[:step_size][end,:]))
        ess_df = ess(chn)
        duration = round(MCMCChains.compute_duration(chn)/60; digits=2)

        @info "Model $(model_idx)" duration
        flush(stderr)
        display(acc_rates)
        display(step_sizes)
        display(ess_df)    
        flush(stdout)    

        @suppress_err @save mcmc_fname chn ess_df;
    end
    return true
end

# model_idx = parse(Int64, ARGS[1])

for model_idx in 1:n_models
    ran = main(model_idx);
end




