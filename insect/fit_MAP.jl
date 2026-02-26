# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
dir_idx = parse(Int64, ARGS[1])
genmodel_idx = feasible_idxs[dir_idx]

OUTDIR = mkpath(joinpath(@__DIR__, "output", "data$(dir_idx)")) # output directory

# Load stored MAP if true, otherwise compute afresh
LOAD_MAP = false;

# Fetch packages.
using PEtab, OrdinaryDiffEq, Optim
using JLD2, ProgressMeter, Random, Suppressor

# @load joinpath(@__DIR__, "params.jld2") tuned_params;
@load joinpath(@__DIR__, "data.jld2") all_data;

const DEFAULT_OPT = Optim.Options(iterations = 1000, show_trace = false, show_warnings = false,
                                  allow_f_increases = true, successive_f_tol = 3,
                                  f_reltol = 1e-8, g_tol = 1e-6, x_abstol = 0.0)

# Perform inference for a PEtabODEProblem.
function fit_petab_prob(petab_prob)
    return calibrate_multistart(
        petab_prob, BFGS(linesearch = Optim.BackTracking()), 10; 
        sample_prior=true, options=DEFAULT_OPT
    )
end

# Fits all models sequentially to one dataset.
data = all_data[genmodel_idx];
petab_models = [create_petab_model(model, data, u0) for model in models];
petab_probs = [PEtabODEProblem(pmodel; odesolver=ODESolver(Rodas5P(), verbose=false)) for pmodel in petab_models];
shuffle_perm = randperm(length(petab_probs)) # shuffle probs to get more accurate time estimate

if LOAD_MAP
    @load "$OUTDIR/MAP.jld2" model_fits
    # compute optimised value to properly "initialise" PEtabProb
    nllhs = [petab_prob.nllh(model_fit.xmin) for (petab_prob, model_fit) in zip(petab_probs, model_fits)];
else
    fit_times = zeros(length(petab_probs))
    shuffled_model_fits = @showprogress [
        begin
            t0 = time_ns()
            model_fit = fit_petab_prob(petab_probs[i])
            fit_times[i] = (time_ns() - t0)/1e9     
            model_fit
        end for i in shuffle_perm
    ]
    model_fits = shuffled_model_fits[invperm(shuffle_perm)] # unshuffle
    nllhs = [petab_prob.nllh(model_fit.xmin; prior=false) for (petab_prob, model_fit) in zip(petab_probs, model_fits)];
    @suppress_err @save "$OUTDIR/MAP.jld2" model_fits fit_times nllhs
end

hess_times = zeros(length(petab_probs))
shuffled_MAP_hessians = @showprogress [
    begin
        t0 = time_ns()
        hess = petab_probs[i].hess(model_fits[i].xmin)
        hess_times[i] = (time_ns() - t0)/1e9
        hess
    end for i in shuffle_perm
];
MAP_hessians = shuffled_MAP_hessians[invperm(shuffle_perm)]; # unshuffle

@save "$OUTDIR/MAP_hess.jld2" MAP_hessians hess_times
