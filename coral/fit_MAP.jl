include(joinpath(@__DIR__, "setup.jl"));

using ForwardDiff, Optim

function fit_MAP(model_sym)
    target = target_dict[model_sym]
    nlpd_func(x) = -LogDensityProblems.logdensity(target, x) # negative log posterior density
    opt_sol = optimize(nlpd_func, guess_dict[model_sym], BFGS())
    MAP = opt_sol.minimizer
    hess = ForwardDiff.hessian(nlpd_func, MAP)
    return (
        MAP = MAP,
        hess = hess,
    )
end

model_fits = Dict(
    sym => @timed fit_MAP(sym) for sym in model_syms
);

mkpath(joinpath(@__DIR__, "output"))
fname = joinpath(@__DIR__, "output/MAPs.jld2");
@save fname model_fits