using DataFrames, XLSX
using JLD2, Suppressor
using LogDensityProblems, LogDensityProblemsAD, PreallocationTools
using Distributions, ForwardDiff, LinearAlgebra
using LaTeXStrings

DATA_FNAME = joinpath(@__DIR__, "data.xlsx");
df = DataFrame(XLSX.readtable(DATA_FNAME, "Sheet1", infer_eltypes=true))
times = convert.(Float64, df[!, "Time"])
obs = df[!, "Site 1"]
n_times = length(times)
tmax = 4200


# Analytical solutions take positive parameters as input

function logistic_sol(θpos, t)
    r, K, C0, _ = θpos
    return K * C0 / (C0 + (K - C0) * exp(-r * t))
end


function gompertz_sol(θpos, t)
    r, K, C0, _ = θpos
    return K * exp(log(C0 / K) * exp(-r * t))
end


function richards_sol(θpos, t)
    r, K, C0, β, _ = θpos
    return K / (1 + ((K / C0)^β - 1) * exp(-r * β * t))^(1 / β)
end


# Everything else is defined on unconstrained scale (log10)

struct LogPost{F,D}
    times::Vector{Float64}
    obs::Vector{Float64}
    dim::Int64
    sol_func::F
    prior_dists::Vector{D}
    param_cache::DiffCache{Vector{Float64}, Vector{Float64}}
end


function LogPost(times, obs, sol_func, prior_dists)
    d = length(prior_dists)
    LogPost(times, obs, length(prior_dists), sol_func, prior_dists, DiffCache(zeros(d)))#, zeros(T, n_times))
end


function compute_logprior(prior_dists, θ)
    sum(logpdf(dist, x) for (dist, x) in zip(prior_dists, θ))
end


LogDensityProblems.logdensity(prob::LogPost, θ) = begin
    θpos = get_tmp(prob.param_cache, θ)
    broadcast!(exp10, θpos, θ)
    σ2 = abs2(θpos[end])
    lpdf = sum(logpdf(dist, x) for (dist, x) in zip(prob.prior_dists, θ)) # log prior
    lpdf -= 0.5log(2π*σ2) * length(prob.times)
    for (t, Cobs) in zip(prob.times, prob.obs)
        Csim = prob.sol_func(θpos, t)
        lpdf -= abs2(Csim - Cobs)/(2*σ2)
    end
    return lpdf
end
LogDensityProblems.dimension(prob::LogPost) = prob.dim;

model_syms = [:logistic, :gompertz, :richards]

name_dict = Dict(:logistic => "logistic", :gompertz => "Gompertz", :richards => "Richards'")

r_prior = Normal(-3, 3);
K_prior = Normal(2, 3);
C0_prior = Normal(0, 3);
β_prior = Normal(0, 3);
σ_prior = Normal(0, 1);

psym_dict = Dict(
    :logistic => [:r, :K, :C0, :σ],
    :gompertz => [:r, :K, :C0, :σ],
    :richards => [:r, :K, :C0, :β, :σ]
);

param_labels = [L"r", L"K", L"C(0)", L"\beta", L"\sigma"];
sym2label = Dict(zip([:r, :K, :C0, :β, :σ], param_labels))

nparam_dict = Dict(k => length(v) for (k, v) in psym_dict)

priors_dict = Dict(
    :logistic => [r_prior, K_prior, C0_prior, σ_prior],
    :gompertz => [r_prior, K_prior, C0_prior, σ_prior],
    :richards => [r_prior, K_prior, C0_prior, β_prior, σ_prior]
);

# Guesses based on https://github.com/ProfMJSimpson/SigmoidGrowth/tree/main/MLE
guess_dict = Dict(
    :logistic => log10.([0.002, 80., 2., 10.]),
    :gompertz => log10.([0.002, 80., 2., 10.]),
    :richards => log10.([0.002, 80., 2., 1., 10.])
);

solfunc_dict = Dict(
    :logistic => logistic_sol,
    :gompertz => gompertz_sol,
    :richards => richards_sol,
);

target_dict = Dict(
    sym => begin
        prob = LogPost(times, obs, solfunc, priors_dict[sym])
        ADgradient(:ForwardDiff, prob);
    end for (sym, solfunc) in solfunc_dict
)


# using Logging, LoggingExtras

# nowarn_logger = EarlyFilteredLogger(global_logger()) do log
#     log.level != Logging.Warn
# end

# macro nowarn_load(filename, vars...)
#     quote
#         ($([esc(v) for v in vars]...),) =
#             with_logger(nowarn_logger) do
#                 ($([:(load($(esc(filename)), $(string(v)))) for v in vars]...),)
#             end

#         $(Symbol[v for v in vars])
#     end
# end