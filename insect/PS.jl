include(joinpath(@__DIR__, "../PS_functions.jl"));
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

μ_slab, σ_slab = 0., 2.;
μ_noise, σ_noise = -1, 1;
slab_prior = Normal(μ_slab, σ_slab)
noise_prior = Normal(μ_noise, σ_noise)

n_priors = 25;
esize = 4;
μ_spike = -16;
σ_spike = σ_slab;

μ0, σ0 = -4., 3.;

function interpolate_ss(iter, μ_trg, σ_trg)
    n_inter = n_priors-1
    r = σ_trg/σ0
    σ = exp(log(σ0) + (iter/n_inter)*log(r))
    μ = μ0 + (1-r^(iter/n_inter))/(1-r)*(μ_trg - μ0)
    return Normal(μ, σ)
end

slab_seq = interpolate_ss.(0:(n_priors-1), μ_slab, σ_slab);
spike_seq = interpolate_ss.(0:(n_priors-1), μ_spike, σ_spike);
thres_vec = 0.5 .* (getproperty.(slab_seq, :μ) .+ getproperty.(spike_seq, :μ));

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

function make_ldp(logprior_func::Function, β::Float64)
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
        return ll * β
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

function should_rerun_nuts(
    particles, prev_states, iter, targetinfo;
    min_class_frac = 0.01, min_count = 20, verbose = 0
)
    pop_size = length(particles)
    min_count = max(min_count, floor(Int, min_class_frac * pop_size))
    D = length(prev_states[1])
    thres = thres_vec[targetinfo[1]]

    curr_states = [p.state for p in particles]
    prev_classes = partition_by_thres(prev_states, thres)
    
    for (key, pre_idxs) in prev_classes
        length(pre_idxs) < min_count && continue

        for d in 1:D
            c = corkendall(getindex.(prev_states[pre_idxs], d), getindex.(curr_states[pre_idxs], d))
            if c > 0.5
                if verbose > 0
                    @info "Rerun due to $key" length(pre_idxs) d c
                end
                return true
            end
        end
    end 
    return false
end

μ_init = [fill(μ_slab, length(std_idxs)); fill(μ0, length(ss_idxs)); μ_noise];
σ_init = [fill(σ_slab, length(std_idxs)); fill(σ0, length(ss_idxs)); σ_noise];
init_dist = MvNormal(μ_init, σ_init);
init_sampler = (rng) -> rand(rng, init_dist);

@info "Threads" LinearAlgebra.BLAS.get_num_threads() Threads.nthreads()
flush(stdout)
flush(stderr)

pop_size = 10000; target_ess = 20000; init_pop_size = 40000;
vid_path = mkpath(joinpath(@__DIR__, "imgs/PS0/data$(dir_idx)"));
fname = joinpath(@__DIR__, "output/data$(dir_idx)/PS0.jld2");
run_PS(
    pop_size, target_ess, init_sampler, logprior_funcs, make_ldp, nuts_move, should_rerun_nuts, fname;
    init_stepsize=1e-1, init_pop_size=init_pop_size,
    verbose=1, vid_path=vid_path, make_fig=make_fig,
    parallel=true, pbar_lines=20,
    # parallel=false, 
);


