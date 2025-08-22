# Fetch packages.
using CairoMakie, Catalyst, Combinatorics, DataFrames, Distributions, JLD2, OrdinaryDiffEq, PEtab, ProgressMeter, Optim, StableRNGs
using OptimizationBBO, OptimizationOptimJL

# Generates all potential models.
begin
    t = default_t()
    all_ps = @parameters k2 k3 r d1 d2 d3 K1 K2 K3 σ
    @species E(t) L(t) A(t)
    rxs_base = [
        Reaction(k2, [E], [L]),
        Reaction(k3, [L], [A]),
        Reaction(r, [A], [A, E]),
    ]
    rxs_extra = [
        Reaction(d1, [E], []),
        Reaction(d2, [L], []),
        Reaction(d3, [A], []),
        Reaction(K1, [E], [], [2], []),
        Reaction(K2, [L], [], [2], []),
        Reaction(K3, [A], [], [2], []),
    ]
    models = []
    for rxs in [[], collect(combinations(rxs_extra))...]
        @named rs = ReactionSystem(
            [rxs_base; rxs], t, [E, L, A], 
            [reduce(vcat, [Symbolics.get_variables(rx.rate) ∩ all_ps for rx in [rxs_base; rxs]])...; σ])
        push!(models, complete(rs))
    end
    u0 = [:E => 5.0, :L => 0.0, :A => 0.0]
	t_final = 10.
end

@load "params.jld2" param_sets
@load "data.jld2" all_data

function fit_model(model, data, u0)
    obs = Dict(
        "obs_E" => PEtabObservable(max(0,model.E), 0.01 + model.σ * max(0,model.E)),
        "obs_L" => PEtabObservable(max(0,model.L), 0.01 + model.σ * max(0,model.L)),
        "obs_A" => PEtabObservable(max(0,model.A), 0.01 + model.σ * max(0,model.A))
    )
    p_est = [PEtabParameter(ModelingToolkit.getname(p)) for p in parameters(model)]
    measurements = vcat(
        DataFrame(obs_id = "obs_E", time = data.t, measurement = data.data_E),
        DataFrame(obs_id = "obs_L", time = data.t, measurement = data.data_L),
        DataFrame(obs_id = "obs_A", time = data.t, measurement = data.data_A)
    )
    petab_model = PEtabModel(model, obs, measurements, p_est; speciemap = u0)
    petab_prob = PEtabODEProblem(petab_model)
    petab_fit = calibrate_multistart(petab_prob, BFGS(), 10)
    return (petab_model, petab_fit)
end
# n_models = length(models); n_data = length(all_data);
data_idx = parse(Int64, ARGS[1]);
data = all_data[data_idx];
model_fits = @showprogress [fit_model(model, data, u0) for model in models];
@save "mle_fits/data$(data_idx).jld2" model_fits
