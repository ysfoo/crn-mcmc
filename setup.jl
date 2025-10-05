# Fetch packages.
using Catalyst, Combinatorics
using DataFrames, Distributions, PEtab

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
        Reaction(K1, [E], [E], [2], [1]),
        Reaction(K2, [L], [L], [2], [1]),
        Reaction(K3, [A], [A], [2], [1]),
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

function create_petab_model(model, data, u0)
    obs = Dict(
        "obs_E" => PEtabObservable(max(0,model.E), 0.01 + model.σ * max(0,model.E)),
        "obs_L" => PEtabObservable(max(0,model.L), 0.01 + model.σ * max(0,model.L)),
        "obs_A" => PEtabObservable(max(0,model.A), 0.01 + model.σ * max(0,model.A))
    )
    p_est = [PEtabParameter(
        ModelingToolkit.getname(p);
        lb = 0., ub = Inf,
        prior = ModelingToolkit.getname(p) == :σ ? Normal(-1, 1) : Normal(0, 2),
        prior_on_linear_scale = false
    ) for p in parameters(model)]
    measurements = vcat(
        DataFrame(obs_id = "obs_E", time = data.t, measurement = data.data_E),
        DataFrame(obs_id = "obs_L", time = data.t, measurement = data.data_L),
        DataFrame(obs_id = "obs_A", time = data.t, measurement = data.data_A)
    )
    return PEtabModel(model, obs, measurements, p_est; speciemap = u0)
end