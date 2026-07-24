include(joinpath(@__DIR__, "../SMC_functions.jl"));
include(joinpath(@__DIR__, "../ODE_LDP.jl"));
include(joinpath(@__DIR__, "setup.jl"));

using Distributions, LinearAlgebra, Optim
using JLD2

using ThreadPinning
isslurmjob() = get(ENV, "SLURM_JOBID", "") != ""
isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);

LinearAlgebra.BLAS.set_num_threads(1)

# This script takes one command-line argument, which is the index of `feasible_idxs`.
dir_idx = parse(Int64, ARGS[1])
# dir_idx = 25

@load joinpath(@__DIR__, "data.jld2") all_data;
data = all_data[feasible_idxs[dir_idx]];
n_obs = length(data.t)
data_mat = hcat(data.data_E, data.data_L, data.data_A) # N x 3

rx_sys = models[end]
n_θ = length(parameters(rx_sys))
std_idxs = 1:3 # ODE parameters with standard prior
ss_idxs = 4:9 # ODE parameters with spike-and-slab prior
noise_idx = 10
n_ss = length(ss_idxs)

base_oprob = ODEProblem(rx_sys, u0, (0.0, 10.0), [p => 1. for p in parameters(rx_sys)]);

param_idxs = map((x)->parameter_index(base_oprob, x).idx, parameters(rx_sys))
u0_idxs = map((x)->parameter_index(base_oprob, Initial(x)).idx, unknowns(rx_sys))

ode_params!(buf, θ) = for i in 1:n_θ
    buf[i] = exp10(θ[i]) 
end;

param_labels = [
    L"\lambda_{EL}", L"\lambda_{LA}", L"\rho", 
    L"\delta_E", L"\delta_L", L"\delta_A", 
    L"\kappa_E", L"\kappa_L", L"\kappa_A", L"\sigma"
];
sym2label = Dict(zip(Symbol.(parameters(models[end])), param_labels))

function make_fig(particles, iter)
    pop_size = length(particles)
    f = plot_pairs(
        getproperty.(particles, :state);
        title="Iteration $iter",
        figsize=(1000, 1000), skip_upper=true,
        axis_kwargs=(;), hist_axis_kwargs=(; yscale=identity),
        scatter_kwargs=(; markersize=4, alpha=clamp(1000 / pop_size * 0.3, 0.01, 0.5)),
        hist_kwargs=(bins=50,),
    )
    Box(f[ss_idxs, ss_idxs], color=(:green, 0.2), strokevisible=false)

    ps = parameters(models[end])
    d = n_θ
    idx = 0
    for (i1, p1) in enumerate(ps) # which row
        for (i2, p2) in enumerate(ps) # which column
            if i1 < i2
                continue
            end
            idx += 1
            ax = f.content[idx]
            if i1 == i2
                ax.yaxisposition = :right
                ax.yticklabelsize = 14
                ax.yticklabelpad = 0.5
                ax.yticksvisible = true
                ax.yticklabelsvisible = true
            elseif i2 ∈ [1, d]
                ax.yaxisposition = i2 == 1 ? :left : :right
                ax.ylabel = L"\log_{10} %$(sym2label[Symbol(p1)])"
                ax.ylabelsize = 18
                ax.yticks = WilkinsonTicks(6; k_min = 3, k_max=6)
                ax.yticklabelsize = 14
                ax.yticksvisible = true
                ax.yticklabelsvisible = true                
            else
                ax.yticksvisible = false
                ax.yticklabelsvisible = false
            end
            
            if i1 ∈ [d]
                ax.xaxisposition = i1 == 1 ? :top : :bottom
                ax.xlabel = L"\log_{10} %$(sym2label[Symbol(p2)])"
                ax.xlabelsize = 18
                ax.xticks = WilkinsonTicks(6; k_min = 3, k_max=6)
                ax.xticklabelrotation = π/4
                ax.xticklabelsize = 14
            else
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
            end
        end
    end
    colgap!(f.layout, -8)
    return f
end

# esize = 4;
μ_noise, σ_noise = -1, 1;
μ_slab, σ_slab = 0., 2.;
μ_spike = -16; σ_spike = σ_slab;
noise_prior = Normal(μ_noise, σ_noise)
slab_prior = Normal(μ_slab, σ_slab)
spike_prior = Normal(μ_spike, σ_spike)


function interpolate_ss(idx, μ0, σ0, μ_trg, σ_trg, temper_prior)
    if n_priors == 1
        return temper_prior ? Normal(μ0, σ0) : Normal(μ_trg, σ_trg)
    end
    n_inter = n_priors-1
    r = σ_trg/σ0
    σ = exp(log(σ0) + (idx/n_inter)*log(r))
    μ = μ0 + (1-r^(idx/n_inter))/(1-r)*(μ_trg - μ0)
    return Normal(μ, σ)
end

function sq_hellinger(dist1, dist2)
    avg_var = (dist1.σ^2 + dist2.σ^2) / 2
    overlap = sqrt(dist1.σ*dist2.σ/avg_var) * exp(-(dist1.μ - dist2.μ)^2/(8avg_var))
    return 1 - overlap
end

function make_ldp(logprior_func::Function, β::Float64, loglike_offset::Function=(θ)->0.)
    function loglike_func(sol, θ::AbstractVector{T}) where T
        σ = exp10(θ[noise_idx])
        ll = T(-n_u*n_obs*log(2π)/2)
        for i in 1:n_obs
            for j in 1:n_u
                y_obs = data_mat[i,j]
                y_pred = sol.u[i][j]
                s = 0.01 + σ*max(0., y_pred)
                ll -= ((y_obs - y_pred) / s)^2 / 2 + log(s)
            end
        end
        return (ll + loglike_offset(θ)) * β
    end
    return OrdinaryDiffEqLDP(
        base_oprob, param_idxs, ode_params!, logprior_func, loglike_func, n_θ;
        solve_kwargs = (saveat=data.t, verbose=false),
    )
end

function partition_by_thres(states::AbstractVector, thres::Float64)
    classes = Dict{BitVector, Vector{Int}}()
    for (i, θ) in enumerate(states)
        key = BitVector(θ[j] > thres for j in ss_idxs)
        push!(get!(classes, key, Int[]), i)
    end
    return classes
end

function no_rerun(particles, prev_states, iter, npass, targetinfo; verbose = 0)
    return false
end

function rerun_2(particles, prev_states, iter, npass, targetinfo; verbose = 0)
    return npass < 2
end

function rerun_by_kendall!(
    particles, prev_states, iter, npass, targetinfo;
    min_count = 20, verbose = 0, update_stepsize=false
)
    pop_size = length(particles)
    if update_stepsize
        mean_acc_rate = mean(p.info.curr_acc_rate for p in particles)
        new_stepsize = particles[1].stepsize * 1.5^((mean_acc_rate-0.8)/0.2)
        for i in 1:pop_size
            particle = particles[i]
            particles[i] = (@set particle.stepsize = new_stepsize)
        end
    end

    D = length(prev_states[1])
    thres = thres_vec[targetinfo[1]]

    curr_states = [p.state for p in particles]
    prev_classes = partition_by_thres(prev_states, thres)

    for (key, pre_idxs) in prev_classes
        n_idxs = length(pre_idxs)
        n_idxs < min_count && continue
        ties = collect(n for (_, n) in countmap(prev_states[pre_idxs]) if n > 1)
        pairs = n_idxs*(n_idxs-1)÷2
        tied_pairs = sum(t*(t-1)÷2 for t in ties; init=0)
        # cor_thres = kendall_qt(n_idxs, 1e-4, ties) + 0.2
        cor_thres = 0.2 + 0.8 * sqrt(min_count / n_idxs)
        for d in 1:D
            c = corkendall(getindex.(prev_states[pre_idxs], d), getindex.(curr_states[pre_idxs], d))
            if c > cor_thres
                if verbose > 0
                    @info "Rerun due to $key, dimension $d" length(pre_idxs) (n_idxs, tied_pairs) (c, cor_thres)
                end
                return true
            end
        end
    end
    return false
end

### Configure run

μ0, σ0 = -8., 4.;
n_priors = 1;

# A. Temper likelihood and prior, adapt # NUTS iterations, particle-specific step size
# run_str = "SMC662"
# temper_prior = true;
# move_func = nuts_move;
# rerun_func = rerun_by_kendall!;
# ldp_builder = (logprior_func, β) -> make_ldp(logprior_func, β, θ -> final_logprior_func(θ) - logprior_funcs[begin](θ));

# B. Temper likelihood only, adapt # NUTS iterations, particle-specific step size
# run_str = "SMC262"
# temper_prior = false;
# move_func = nuts_move;
# rerun_func = rerun_by_kendall!;
# ldp_builder = make_ldp;

# C. Temper likelihood and prior, 2 * 6 NUTS iterations, particle-specific step size
# run_str = "SMC602"
# temper_prior = true;
# move_func = (rng, particle, target) -> nuts_move(rng, particle, target; n_nuts=6);
# rerun_func = (particles, prev_states, iter, npass, targetinfo; verbose=0) -> npass < 2;
# ldp_builder = (logprior_func, β) -> make_ldp(logprior_func, β, θ -> final_logprior_func(θ) - logprior_funcs[begin](θ));

# D. Temper likelihood and prior, adapt # NUTS iterations, shared step size
run_str = "SMC660"
temper_prior = true;
move_func = (rng, particle, target) -> nuts_move(rng, particle, target; adapt_stepsize_func=no_adapt_func);
rerun_func = (
    (particles, prev_states, iter, npass, targetinfo; verbose=0) 
    -> rerun_by_kendall!(particles, prev_states, iter, npass, targetinfo; verbose, update_stepsize=true)
);
ldp_builder = (logprior_func, β) -> make_ldp(logprior_func, β, θ -> final_logprior_func(θ) - logprior_funcs[begin](θ));

### End configure

slab_seq = interpolate_ss.(0:(n_priors-1), μ0, σ0, μ_slab, σ_slab, temper_prior);
spike_seq = interpolate_ss.(0:(n_priors-1), μ0, σ0, μ_spike, σ_spike, temper_prior);
thres_vec = 0.5 .* (getproperty.(slab_seq, :μ) .+ getproperty.(spike_seq, :μ))
logprior_funcs = [
    begin
        mix_prior = MixtureModel([slab, spike])
        dists = [
            fill(slab_prior, length(std_idxs));
            fill(mix_prior,  length(ss_idxs));
            noise_prior
        ]
        (θ) -> sum(logpdf(dist, val) for (dist, val) in zip(dists, θ))
    end for (slab, spike) in zip(slab_seq, spike_seq)
];

init_dists = begin
    mix_prior = MixtureModel([slab_seq[1], spike_seq[1]])
    [
        fill(slab_prior, length(std_idxs));
        fill(mix_prior,  length(ss_idxs));
        noise_prior
    ];
end
init_sampler = (rng) -> rand.(Ref(rng), init_dists);

final_ss_prior = MixtureModel([slab_prior, spike_prior])
final_dists = [
    fill(slab_prior, length(std_idxs));
    fill(final_ss_prior, length(ss_idxs));
    noise_prior
]
final_logprior_func(θ) = sum(logpdf(dist, val) for (dist, val) in zip(final_dists, θ))

β_thres_zero(j) = 0.

# pop_size = 1000; target_ess = 800; init_pop_size = 1000;
pop_size = 5000; target_ess = 4000; init_pop_size = 5000;

fname = joinpath(@__DIR__, "output/data$(dir_idx)/$(run_str).jld2");
# vid_path = joinpath(@__DIR__, "imgs/$(run_str)/data$(dir_idx)");
# mkpath(vid_path)
run_SMC(
    pop_size, target_ess, init_sampler, logprior_funcs, ldp_builder, move_func, rerun_func, fname;
    init_stepsize=1e-1, init_pop_size=init_pop_size,
    β_thres_func=β_thres_zero, verbose=1, #vid_path=vid_path, make_fig=make_fig,
    parallel=true, pbar_lines=20,
);
