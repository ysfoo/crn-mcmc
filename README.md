# WIP: Bayesian model selection of chemical reaction networks

## Workflow

1. `generate_datasets.jl`: Defines 64 possible CRN models for the population dynamics of a hypothetical insect (egg/larva/adult life stages). Tune ground truth parameters for each model to reach a target state at some time, and generate a dataset for each parametrised model.

2. `fit_mle.jl`: Fits all models to a dataset, whose index (int) is specified as a command-line argument. Results are stored in a directory named `mle_fits/`.