# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));

# Fetch packages.
using CairoMakie, Distributions, JLD2, LineSearches, ProgressMeter, Optim, StableRNGs
using OptimizationBBO, OptimizationOptimJL

const RNG = StableRNG(1);

function obj_osc(sol, f, f2sum)
    return [begin
        s = getindex.(sol.u, d)
        sum(abs2, valid_conv(f, s)) / sum_valid_conv(f.*f, s.*s) * f2sum
    end for d in eachindex(sol.u[1])]
end

# Optimise groud truth parameter values to exhibit certain behaviour:
# - Trajectories end near `u_trg`,
# - Trajectory curvature is not too large,
# - Parameters are penalised by negative log density of inverse gamma
function tune_params(model, u0, t_final, u_trg, λ1=1e-3, λ2=1e-3, n_save=101)
    p0 = [p => 1e-2 + rand(RNG) for p in parameters(model)]
    n_p = length(p0)
    oprob_base = ODEProblem(model, u0, t_final, p0)
    set_p = ModelingToolkit.setp_oop(oprob_base, collect(parameters(model)))    
    saveat = range(0, t_final, n_save)
    Δt = t_final/(n_save-1)

    function sum_curvature(sol)
        sol_mat = reduce(hcat, sol.u)
        return sum(abs2, diff(diff(sol_mat, dims=2), dims=2))
    end

    function invΓ_obj(p, a=4., b=3.)
        return (a+1)*log(p) + b/p
    end

    function parameter_suitability(p, (oprob_base, set_p, u_trg))
        mtk_p = set_p(oprob_base, p)
        oprob = remake(oprob_base; p=mtk_p)
        sol = solve(oprob; saveat, verbose = false, maxiters = 10000)
        SciMLBase.successful_retcode(sol) || return Inf        
        try 
            return (
                sum(abs2 ∘ log, sol.u[end] ./ u_trg)
                + λ1*sum_curvature(sol) / Δt^3
                # + λ1*exp(0.5*sum(abs2, sol.u[div(n_save,2)] .- sol.u[end]))
                + λ2*sum(invΓ_obj, p)
            )
        catch e
            e isa DomainError && return Inf
            throw(e)
        end
    end

    opt_func = OptimizationFunction(parameter_suitability, Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(opt_func, last.(p0)[1:end-1], (oprob_base, set_p, u_trg);
        lb = [fill(1e-2, n_p-1)...], ub = [fill(10.0, n_p-1)...])
    opt_sol = solve(opt_prob, BFGS(linesearch = LineSearches.Backtracking()))

    # opt_prob = OptimizationProblem(parameter_suitability, last.(p0)[1:end-1], (oprob_base, set_p, u_trg);
    #     lb = [fill(1e-2, n_p-1)...], ub = [fill(100.0, n_p-1)...])
    # opt_sol = solve(opt_prob, BBO_adaptive_de_rand_1_bin_radiuslimited())
    
    return opt_sol
    # return [p => p_val for (p, p_val) in zip(parameters(model), [opt_sol.u..., 0.05])]
end

# bbo_sols = @showprogress [tune_params(model, u0, 10., [3., 2., 1.]) for model in models[1:64]];
# param_sets = [
#     [p => p_val for (p, p_val) in zip(parameters(model), [opt_sol.u..., 0.05])]
#     for (model, opt_sol) in zip(models, bbo_sols)
# ];

bfgs_sols = @showprogress [tune_params(model, u0, t_final, [3., 2., 1.]) for model in models];
param_sets = [
    [p => p_val for (p, p_val) in zip(parameters(model), [opt_sol.u..., 0.05])]
    for (model, opt_sol) in zip(models, bfgs_sols)
];
getproperty.(getproperty.(bfgs_sols, :original), :termination_code)

# Plot all tuned parameters.
begin
    param_vecs = Dict(p => Float64[] for p in parameters(models[end]));
    for pset in param_sets
        for (p, pval) in pset
            push!(param_vecs[p], pval)
        end
    end
    f = Figure();
    for i in 1:9
        p = parameters(models[end])[i] 
        ax = Axis(
            f[div(i-1, 3) + 1, mod(i-1, 3) + 1], 
            xlabel=string(p), xscale=log10
        );
        hist!(param_vecs[p], bins=logrange(extrema(param_vecs[p])..., 20))
    end
    display(f)
end

# Generate data.
function generate_data(model, u0, t_final, ps; n = 41, σ = 0.05)
    oprob = ODEProblem(model, u0, t_final, ps)
    sol = solve(oprob; saveat = range(0, t_final, n))
    data_E = [rand(RNG, Normal(max(0,E), 0.01+max(0,E)*σ)) for E in sol[:E]]
    data_L = [rand(RNG, Normal(max(0,L), 0.01+max(0,L)*σ)) for L in sol[:L]]
    data_A = [rand(RNG, Normal(max(0,A), 0.01+max(0,A)*σ)) for A in sol[:A]]
    return (        
        data_E = data_E,
        data_L = data_L,
        data_A = data_A,
        t = sol.t,
    )
end
all_data = [generate_data(model, u0, t_final, ps) for (model,ps) in zip(models, param_sets)];

# Plot data.
function make_plot!(model, u0, ps, data)
    oprob = ODEProblem(model, u0, data.t[end], ps)
    sol = solve(oprob)
    for d in eachindex(sol.u[1])
        lines!(sol.t, getindex.(sol.u, d), alpha=0.7)
        scatter!(data.t, data[d])
    end
end
function make_plot!(idx, models, u0, param_sets, all_data)
    make_plot!(models[idx], u0, param_sets[idx], all_data[idx])
end

begin
    f = Figure(size=(1200, 1200));
    @showprogress for (i, model) in enumerate(models)        
        ax = Axis(f[div(i-1, 8) + 1, mod(i-1, 8) + 1]);
        make_plot!(i, models, u0, param_sets, all_data)
    end
    display(f)
end

@save "params.jld2" param_sets
@save "data.jld2" all_data
