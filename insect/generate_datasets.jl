# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));

# Fetch packages.
using CairoMakie, Distributions, JLD2, LinearAlgebra, OrdinaryDiffEq, ProgressMeter, Random, StableRNGs
using ForwardDiff, NonlinearSolve, Optim, OptimizationBBO, OptimizationOptimJL, PreallocationTools

function gamma_obj(p, a=2., b=2.)
    return -(a-1)*log(p) + b*p
end

function calc_objs(p, sol, u_trg, du, t_end)
    return [
        sum(gamma_obj, p),
        log1p((10norm(du))^10),
        log1p((10norm(sol(t_end) .- u_trg))^10),
    ]
end


# Optimise groud truth parameter values to exhibit certain behaviour...
function tune_params(model, u0, t_end, u_trg; rng=Random.default_rng(), n_start=5)
    p0 = [p => exp10(rand(rng) - 1) for p in parameters(model)]
    n_p = length(p0)
    oprob_base = ODEProblem(model, u0, t_end, p0)
    set_p = ModelingToolkit.setp_oop(oprob_base, collect(parameters(model)))

    condition(u, t, integrator) = maximum(u) > 100
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)

    function sol_diff(sol, t0, t1)
        return norm(sol(t0) .- sol(t1))
    end

    function parameter_suitability(p, (oprob_base, set_p, u_trg))
        mtk_p = set_p(oprob_base, p)
        oprob = remake(oprob_base; p=mtk_p)
        sol = solve(oprob; verbose = false, maxiters = 10000, callback=cb)
        if !SciMLBase.successful_retcode(sol) || maximum(sol.t) < t_end
            return Inf        
        end
        du = similar(sol.u[end])
        oprob.f(du, sol.u[end], mtk_p.tunable, 0)
        return sum(calc_objs(p, sol, u_trg, du, t_end))
    end

    opt_func = OptimizationFunction(parameter_suitability, Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(opt_func, last.(p0)[1:end-1], (oprob_base, set_p, u_trg);
        lb = [fill(0.1, n_p-1)...], ub = [fill(10., n_p-1)...])

    p0s = [
        begin
            p0 = [p => exp10(rand(rng) - 1) for p in parameters(model)]
            while !isfinite(parameter_suitability(last.(p0)[1:end-1], (oprob_base, set_p, u_trg)))
                p0 = [p => exp10(rand(rng) - 1) for p in parameters(model)]
            end
            p0
        end for _ in 1:n_start
    ]
    function prob_func(opt_prob, i, repeat)
        remake(opt_prob, u0 = p0s[i])
    end
    ensembleprob = EnsembleProblem(opt_prob; prob_func)

    opt_sol = solve(
        ensembleprob, BFGS(linesearch = Optim.BackTracking()), EnsembleThreads();
        trajectories = n_start
    )
    best = argmin((i)->opt_sol[i].objective, eachindex(opt_sol))    
    return opt_sol[best]
end

function check_stability(oprob, p, u0)
    du = similar(u0)
    du_cache = DiffCache(du)
    obj_func(u, p) = begin
        du_tmp = get_tmp(du_cache, u)
        oprob.f(du_tmp, u, p, 0)
        return abs2.(du_tmp) .+ clamp.(abs2.(u .- u0) .- 0.01, 0, Inf)
    end
    prob = NonlinearProblem(obj_func, u0, p)
    sol = solve(prob)
    
    ode_func(u) = begin
        du_tmp = get_tmp(du_cache, u)
        oprob.f(du_tmp, u, p, 0)
        return du_tmp
    end

    return all(sol.u .> 1.), norm(ode_func(sol.u)) < 1e-6, all(real.(eigvals(ForwardDiff.jacobian(ode_func, sol.u))) .< 0)
end

t_end = 10.
u_trg = [2., 3., 4.];
NOISE_LVL = 0.05;


opt_sols = Dict{Int,SciMLBase.OptimizationSolution}();
tuned_params = Dict{Int,Vector{Pair{Any,Float64}}}();
@showprogress for model_idx in feasible_idxs
    model = models[model_idx]
    opt_sol = tune_params(model, u0, t_end, u_trg; rng=StableRNG(model_idx));
    tuned = [p => p_val for (p, p_val) in zip(parameters(model), [opt_sol.u..., NOISE_LVL])]
    opt_sols[model_idx] = opt_sol
    tuned_params[model_idx] = tuned

    oprob_base = ODEProblem(model, u0, t_end, tuned);
    sol = solve(oprob_base; maxiters = 10000);

    fig = Figure()
    ax = Axis(
        fig[1,1],
        limits=((-0.05,10.05),nothing), 
        # xscale=Makie.pseudolog10, xticks=[0,2,5,10,20,50,100,200,500,1000],
        xlabel="Time", ylabel="Population size (a.u.)", title="Data-generating model $(model_idx)"
    )
    
    for d in eachindex(sol.u[1])
        lines!(sol.t, getindex.(sol.u, d), alpha=0.7)
        scatter!([t_end], u_trg[[d]])
    end
    save(joinpath(@__DIR__, "imgs/data_trajs/model$(model_idx).png"), fig)
    # display(fig)

    oprob = remake(oprob_base; tspan=(0, 1000));
    sol = solve(oprob; maxiters = 10000);
    pos_eq, zero_norm, neg_eigvals = check_stability(oprob, oprob.p.tunable, sol.u[end])
    if !(pos_eq && zero_norm && neg_eigvals)
        @info model_idx pos_eq zero_norm neg_eigvals oprob.p
    end
end

# Plot all tuned parameters.
begin
    param_vecs = Dict(p => Float64[] for p in parameters(models[end]));
    for tuned in values(tuned_params)
        for (p, pval) in tuned
            push!(param_vecs[p], pval)
        end
    end
    f = Figure();
    for i in 1:9
        p = parameters(models[end])[i] 
        ax = Axis(
            f[fld1(i, 3), mod1(i, 3)], 
            xlabel=string(p),
        );
        hist!(param_vecs[p], bins=20)
    end
    # display(f)
    save(joinpath(@__DIR__, "imgs/all_params.png"), f)
end

# Generate data.
function generate_data(model, u0, t_end, ps; n = 41, σ = 0.05, rng = Random.default_rng())
    oprob = ODEProblem(model, u0, t_end, ps)
    sol = solve(oprob; saveat = range(0, t_end, n))
    data_E = [rand(rng, Normal(max(0,E), 0.01+max(0,E)*σ)) for E in sol[:E]]
    data_L = [rand(rng, Normal(max(0,L), 0.01+max(0,L)*σ)) for L in sol[:L]]
    data_A = [rand(rng, Normal(max(0,A), 0.01+max(0,A)*σ)) for A in sol[:A]]
    return (        
        data_E = data_E,
        data_L = data_L,
        data_A = data_A,
        t = sol.t,
    )
end
all_data = Dict(
    model_idx => generate_data(models[model_idx], u0, t_end, ps; rng = StableRNG(model_idx)) 
    for (model_idx, ps) in pairs(tuned_params)
);

# Plot data.
function make_plot!(model, u0, ps, data)
    oprob = ODEProblem(model, u0, data.t[end], ps)
    sol = solve(oprob)
    for d in eachindex(sol.u[1])
        lines!(sol.t, getindex.(sol.u, d), alpha=0.8)
        scatter!(data.t, data[d], alpha=0.8, markersize=6)
    end
end
function make_plot!(idx, models, u0, tuned_params, all_data)
    make_plot!(models[idx], u0, tuned_params[idx], all_data[idx])
end

begin
    f = Figure(size=(600, 1080));
    @showprogress for (i, model_idx) in enumerate(feasible_idxs)
        ax = Axis(f[fld1(i, 5), mod1(i, 5)], limits=(nothing, (-0.2, 5)));
        make_plot!(model_idx, models, u0, tuned_params, all_data)
    end
    # display(f)
    save(joinpath(@__DIR__, "imgs/all_data.png"), f)
end

@save "params.jld2" tuned_params;
@save "data.jld2" all_data;
