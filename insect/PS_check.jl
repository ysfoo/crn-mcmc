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
# dir_idx = parse(Int64, ARGS[1])
dir_idx = 25

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

n_priors = 20;
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

function get_mode_counts(states, thres, Ws=ones(length(states)))
    lookup = zeros(Int, 2^n_ss)
    for (i, elems) in enumerate(combinations(1:n_ss))
        idx = 0
        for elem in elems
            idx |= (1 << (elem - 1))
        end
        lookup[idx + 1] = i
    end

    counts = zeros(2^n_ss)
    for (state, W) in zip(states, Ws)
        idx = 0
        for j in 1:n_ss
            if state[ss_idxs[j]] > thres
                idx |= (1 << (j - 1))
            end
        end
        counts[lookup[idx + 1]] += W
    end
    return counts
end

μ_init = [fill(μ_slab, length(std_idxs)); fill(μ0, length(ss_idxs)); μ_noise];
σ_init = [fill(σ_slab, length(std_idxs)); fill(σ0, length(ss_idxs)); σ_noise];
init_dist = MvNormal(μ_init, σ_init);
init_sampler = (rng) -> rand(rng, init_dist);

## Load bridge sampling results

using BridgeSampling
BS_logZvec = [
    begin
        fname = joinpath(@__DIR__, "output/data$(dir_idx)/BS_model$model_idx.jld2")
        @load fname timed_res
        timed_res.value.value
    end for model_idx in 1:n_models
];
BS_ps = exp.(BS_logZvec .- logsumexp(BS_logZvec));
ps = parameters(models[end]);
BS_marg_ps = [sum(pval for (i, pval) in enumerate(BS_ps) if any(parameters(models[i]) .=== p)) for p in ps[4:9]]

## Load PS results

convert_HMS(s) = string(floor(Int, s÷3600), ":", lpad(floor(Int, s%3600÷60), 2, '0'), ":", lpad(floor(Int, s%60), 2, '0'));

# pop_size = 1000; target_ess = 500; init_pop_size = 2000;
# pop_size = 1000; target_ess = 1000; init_pop_size = 2000;

vid_path = mkpath(joinpath(@__DIR__, "imgs/PS1/data$(dir_idx)"));

fname = joinpath(@__DIR__, "output/data$(dir_idx)/PS1.jld2");

@load fname all_particles iter targetinfos logZs all_logws smc_times;
target_ess = 2000;
init_pop_size = length(first(all_particles))
pop_size = length(last(all_particles))

iter
convert_HMS(sum(smc_times))
sum(ps -> ps[1].n_nuts, all_particles[2:end])

compute_ess(reduce(vcat, all_logws[1:end-1]))
compute_ess(reduce(vcat, all_logws))

begin
    f = Figure()
    ax = Axis(f[1,1])
    scatterlines!(0:iter, first.(targetinfos))
    # ax = Axis(f[2,1], yscale=log10, limits=(nothing, (1e-6, 2.)))
    ax = Axis(f[2,1])
    scatterlines!(0:iter, last.(targetinfos))
    display(f)
end

## Compare component proportions

using SpecialFunctions
mad_func(N, p) = p*(1-p) == 0 ? 0 : exp(floor(N*p)*log(p)+(N-floor(N*p))*log(1-p)+logabsbinomial(N-1, floor(Int, N*p))[1])

# Initial
M = 64
init_ps = get_mode_counts(getproperty.(all_particles[1], :state), thres_vec[1]) ./ init_pop_size;
0.5sum(abs, fill(1/M, M) .- init_ps)
sum(p->p*mad_func(init_pop_size, p), fill(1/M, M))

# Final (reference)
sum(p->p*mad_func(1000, p), BS_ps)

# Final without PS
SMC_ps = get_mode_counts(getproperty.(all_particles[end], :state), thres_vec[end]) ./ pop_size;
0.5sum(abs, BS_ps .- SMC_ps)

# Final with PS
cat_logws = reduce(vcat, all_logws);
cat_Ws = exp.(cat_logws .- logsumexp(cat_logws));
scatter(cat_Ws, markersize=6, alpha=0.1);
PS_ps = get_mode_counts([p.state for ps in all_particles for p in ps], thres_vec[end], cat_Ws);
0.5sum(abs, BS_ps .- PS_ps)

begin
    scatter((1:64), BS_ps)
    scatter!((1:64).-0.2, SMC_ps)
    scatter!((1:64).+0.2, PS_ps)
    display(current_figure())
end



## Check NUTS move for single SMC iteration

tmp_i = 25

all_particles[tmp_i + 1][1].n_nuts

for iter_i in [tmp_i]
    particles = all_particles[iter_i + 1]
    afters = getproperty.(particles, :state)
    befores = [p.state .- p.info.delta for p in particles]

    f = Figure(size=(800, 560))
    for i in 1:6
        i1 = cld(i, 3)
        i2 = mod1(i, 3)
        ax = Axis(f[i1,i2], aspect=DataAspect())
        xs, ys = getindex.(befores,ss_idxs[i]), getindex.(afters,ss_idxs[i])
        a, b = extrema([xs; ys])
        lims = (1.025a - 0.025b, 1.025b - 0.025a)
        xlims!(ax, lims)
        ylims!(ax, lims)
        scatter!(
            xs, ys, 
            alpha=0.6, markersize=4
        )
    end
    Label(f[0,:], "Iteration $(iter_i) NUTS", font=:bold, fontsize=18)
    Label(f[3,:], "Before NUTS", fontsize=16)
    Label(f[1:2,0], "After NUTS", fontsize=16, rotation=π/2)
    display(f)
end

d = 6
begin
    iter_i = tmp_i
    particles = all_particles[iter_i + 1]
    afters = getproperty.(particles, :state)
    befores = [p.state .- p.info.delta for p in particles]
    prior_idx, β = targetinfos[iter_i + 1]
    thres = thres_vec[prior_idx]
    prev_classes = partition_by_thres(befores, thres)

    key = Bool[0, 1, 1, 1, 1, 0]
    # tmp_idxs = prev_classes[key]
    tmp_idxs = 1:length(particles)
    
    befores = getindex.(befores[tmp_idxs], d)
    afters = getindex.(afters[tmp_idxs], d)
    stepsizes = getproperty.(particles[tmp_idxs], :stepsize)
    accrates = [p.info.agg_acc_rate for p in particles[tmp_idxs]]

    f = Figure(size=(720,900))
    ax = Axis(f[1,1], xlabel="State before NUTS", ylabel="State after NUTS")
    scatter!(befores, afters, alpha=0.7, markersize=6)
    autolimits!(ax)
    limits!(ax.finallimits[])
    lines!([-100, 100], [-100, 100], linestyle=:dash, color=:black, alpha=0.7)
    lines!([-100, 100], fill(thres, 2), linestyle=:dot, color=:grey30)
    lines!(fill(thres, 2), [-100, 100], linestyle=:dot, color=:grey30)

    ax = Axis(f[1,2], xlabel="State before NUTS", ylabel="NUTS step size", limits=(nothing, (0, nothing)))
    scatter!(befores, stepsizes, alpha=0.3, markersize=6)

    ax = Axis(f[3,1], xlabel="State difference", ylabel="Acceptance rate", limits=(nothing, (-0.02, 1.02)))
    scatter!(afters .- befores, accrates, alpha=0.3, markersize=6)

    ax = Axis(f[2,1], xlabel="State difference", ylabel="NUTS step size", limits=(nothing, (0, nothing)))
    scatter!(afters .- befores, stepsizes, alpha=0.6, markersize=6)

    ax = Axis(f[2,2], xlabel="State before NUTS", ylabel="Acceptance rate", limits=(nothing, (-0.02, 1.02)))
    scatter!(befores, accrates, alpha=0.3, markersize=6)

    ax = Axis(f[3,2], xlabel="NUTS steps size", ylabel="Acceptance rate", limits=((0, nothing), (-0.02, 1.02)))
    scatter!(stepsizes, accrates, alpha=0.6, markersize=6)
    
    display(current_figure())    
    @info (iter_i, d, key) length(tmp_idxs) thres_vec[prior_idx] corkendall(befores, afters)
end

scatter([p.stepsize for p in all_particles[end]], [p.info.agg_acc_rate for p in all_particles[end]])



## Plot states before/after NUTS

figs = Figure[];
@showprogress for iter_i in 1:iter
    particles = all_particles[iter_i + 1]
    afters = getproperty.(particles, :state)
    befores = [p.state .- p.info.delta for p in particles]

    f = Figure(size=(800, 560))
    for i in 1:6
        i1 = cld(i, 3)
        i2 = mod1(i, 3)
        ax = Axis(f[i1,i2], aspect=DataAspect())
        xs, ys = getindex.(befores,ss_idxs[i]), getindex.(afters,ss_idxs[i])
        a, b = extrema([xs; ys])
        lims = (1.025a - 0.025b, 1.025b - 0.025a)
        xlims!(ax, lims)
        ylims!(ax, lims)
        scatter!(
            xs, ys, 
            alpha=0.6, markersize=4
        )
    end
    Label(f[0,:], "Iteration $(iter_i) NUTS", font=:bold, fontsize=18)
    Label(f[3,:], "Before NUTS", fontsize=16)
    Label(f[1:2,0], "After NUTS", fontsize=16, rotation=π/2)
    push!(figs, f)
    save(joinpath(vid_path, "cor_iter$(iter_i).png"), f, px_per_unit=4)
end


## NUTS trends over SMC iterations

begin
    f = Figure(size=(720, 600))

    ax = Axis(
        f[1,1], yscale=log10, limits=((-1, iter+1), nothing),
        ylabel="NUTS step size", xlabel="SMC iteration",
    )
    stepsizes = map((particles) -> map((p) -> p.stepsize, particles), all_particles);
    xs = 0:iter
    scatterlines!(xs, median.(stepsizes), alpha=0.6);
    rangebars!(xs, quantile.(stepsizes, 0.05), quantile.(stepsizes, 0.95), alpha=0.8);

    ax = Axis(
        f[2,1], limits=((-1, iter+1), (-0.02, 1.02)),
        ylabel="NUTS average acceptance rate", xlabel="SMC iteration",
    )
    accrates = map((particles) -> map((p) -> p.info.agg_acc_rate, particles), all_particles[2:end])
    xs = 1:iter
    scatterlines!(xs, median.(accrates), alpha=0.6);
    rangebars!(xs, quantile.(accrates, 0.05), quantile.(accrates, 0.95), alpha=0.8);

    display(f)
end

## Playground

# run_PS(
#     pop_size, target_ess, init_sampler, logprior_funcs, make_ldp, nuts_move, should_rerun_nuts, fname;
#     init_stepsize=1e-1, init_pop_size=init_pop_size,
#     verbose=1, vid_path=vid_path, make_fig=make_fig,
#     parallel=true, pbar_lines=20,
#     # parallel=false, 
# );

# # Approximate critical value of Kendall correlation coefficient
# kendall_qt(n, sig) = begin
#     v = 2 * (2*n + 5) / (9 * n * (n - 1))
#     q = quantile(Normal(), 1 - sig)
#     q*sqrt(v)
# end 

# kendall_qt(20, 1e-3)