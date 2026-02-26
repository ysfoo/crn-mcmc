# Fetch packages.
using Catalyst, Combinatorics
using DataFrames, Distributions, PEtab
using Logging, LoggingExtras

nowarn_logger = EarlyFilteredLogger(global_logger()) do log
    log.level != Logging.Warn
end

macro nowarn_load(filename, vars...)
    quote
        ($([esc(v) for v in vars]...),) =
            with_logger(nowarn_logger) do
                ($([:(load($(esc(filename)), $(string(v)))) for v in vars]...),)
            end

        $(Symbol[v for v in vars])
    end
end

# Generates all potential models.
begin
    t = default_t()
    all_ps = @parameters λ1 λ2 ρ δ1 δ2 δ3 κ1 κ2 κ3 σ
    @species E(t) L(t) A(t)
    rxs_base = [
        Reaction(λ1, [E], [L]),
        Reaction(λ2, [L], [A]),
        Reaction(ρ, [A], [A, E]),
    ]
    rxs_extra = [
        Reaction(δ1, [E], []),
        Reaction(δ2, [L], []),
        Reaction(δ3, [A], []),
        Reaction(κ1, [E], [E], [2], [1]),
        Reaction(κ2, [L], [L], [2], [1]),
        Reaction(κ3, [A], [A], [2], [1]),
    ]
    models = []
    for rxs in collect(combinations(rxs_extra))
        @named rs = ReactionSystem(
            [rxs_base; rxs], t, [E, L, A], 
            [reduce(vcat, [Symbolics.get_variables(rx.rate) ∩ all_ps for rx in [rxs_base; rxs]])...; σ])
        push!(models, complete(rs))
    end
    n_extra = length(rxs_extra);
    n_models = length(models);

# Integer combinations
combs = combinations(collect(1:n_extra));
rx_boolmat = [Int(rx ∈ comb) for comb in combs, rx in 1:n_extra] # 2^R by R
nparams = length.(combs) .+ 4

# For ODE simulaion
u0 = [:E => 0.0, :L => 0.0, :A => 3.0]
n_d = length(u0)
t_end = 10.
end

function model_is_feasible(model)
    syms = Symbol.(parameters(model))
    return any(s ∈ syms for s in [:κ1, :κ2, :κ3]) && any(s ∈ syms for s in [:δ3, :κ3])
end

feasible_idxs = findall(model_is_feasible, models);
n_feasible = length(feasible_idxs)

# function create_petab_model(model, data, u0)
#     obs = [
#         PEtabObservable("obs_E", max(0,model.E), 0.01 + model.σ * max(0,model.E)),
#         PEtabObservable("obs_L", max(0,model.L), 0.01 + model.σ * max(0,model.L)),
#         PEtabObservable("obs_A", max(0,model.A), 0.01 + model.σ * max(0,model.A))
#     ]
#     p_est = [PEtabParameter(
#         ModelingToolkit.getname(p);
#         lb = 0., ub = Inf,
#         prior = ModelingToolkit.getname(p) == :σ ? LogNormal(log(0.1), log(10)) : LogNormal(log(1), log(100)),
#     ) for p in parameters(model)]
#     measurements = vcat(
#         DataFrame(obs_id = "obs_E", time = data.t, measurement = data.data_E),
#         DataFrame(obs_id = "obs_L", time = data.t, measurement = data.data_L),
#         DataFrame(obs_id = "obs_A", time = data.t, measurement = data.data_A)
#     )
#     return PEtabModel(model, obs, measurements, p_est; speciemap = u0)
# end

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

# Latex labels
rx_labels = [
        begin 
        s = string(rx)
        s = s[findfirst(' ', s)+1:end]
        s = Base.replace(s, "-->" => "\\rightarrow")
        s = Base.replace(s, "X" => "X_")
        s = Base.replace(s, "*" => "")
        s
    end for rx in rxs_extra
]