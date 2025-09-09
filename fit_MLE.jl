# This script takes one command-line argument, which is the dataset index.
data_idx = parse(Int64, ARGS[1]);
mkpath(joinpath(@__DIR__, "MLE_fits")) # output directory

# Load stored MLE if true, otherwise compute afresh
LOAD_MLE = true;

# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));

# Fetch packages.
using PEtab, Optim
using JLD2, ProgressMeter, Random, Suppressor

@load "params.jld2" param_sets;
@load "data.jld2" all_data;

# Fits all models sequentially to one dataset.
function fit_petab_prob(petab_prob)
    return calibrate_multistart(petab_prob, BFGS(), 10; sample_prior=true)
end

data = all_data[data_idx];
petab_models = [create_petab_model(model, data, u0) for model in models];
petab_probs = PEtabODEProblem.(petab_models);
shuffle_perm = randperm(length(petab_probs)) # shuffle probs to get more accurate time estimate

if LOAD_MLE
    @load "MLE_fits/data$(data_idx).jld2" model_fits
else
    shuffled_model_fits = @showprogress [fit_petab_prob(petab_probl) for petab_prob in petab_probs[shuffle_perm]];
    model_fits = shuffled_model_fits[invperm(shuffle_perm)] # unshuffle
end

# compute optimised value to properly "initialise" PEtabProb
@time nllhs = [petab_prob.nllh(model_fit.xmin) for (petab_prob, model_fit) in zip(petab_probs, model_fits)];

shuffled_MLE_hessians = @showprogress [
    petab_prob.hess(model_fit.xmin) 
    for (petab_prob, model_fit) in zip(petab_probs[shuffle_perm], model_fits[shuffle_perm])
];
MLE_hessians = shuffled_MLE_hessians[invperm(shuffle_perm)]; # unshuffle

@suppress_err @save "MLE_fits/data$(data_idx).jld2" model_fits MLE_hessians

