# WIP: Bayesian model selection of differential equation models under parameter non-identifiability

This repository hosts the code for reproducing the results of the manuscript "Reliable model selection in the presence of parameter non-identifiability". The following methods for computing model evidence are implemented:

 - Bayesian information criterion (BIC)
 - Laplace importance sampling (Laplace IS)
 - Standard adaptive multiple importance sampling (standard AMIS)
 - Robust adaptive multiple importance sampling (robust AMIS)
 - Bridge sampling

## Coral re-growth example

The results of Section 5.1 are obtained by running the following scripts.

1. TODO

## Insect life-stage example

The results of Section 5.2 are obtained by running the following scripts.

1. `generate_datasets.jl`: Defines 64 possible CRN models for the population dynamics of a hypothetical insect (egg/larva/adult life stages). Tune ground truth parameters for each model to reach a target state at some time, and generate a dataset for each parameterised model.

2. TODO