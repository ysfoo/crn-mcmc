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
# dir_idx = parse(Int64, ARGS[1])
# dir_idx = 25

# @load joinpath(@__DIR__, "data.jld2") all_data;
# data = all_data[feasible_idxs[dir_idx]];
# n_obs = length(data.t)
# data_mat = hcat(data.data_E, data.data_L, data.data_A) # N x 3

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

function partition_by_thres(states::AbstractVector, thres::Float64)
    classes = Dict{BitVector, Vector{Int}}()
    for (i, θ) in enumerate(states)
        mask = BitVector(θ[j] > thres for j in ss_idxs)
        push!(get!(classes, mask, Int[]), i)
    end
    return classes
end

function get_mode_probs(states, thres, Ws=ones(length(states)))
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
    return counts ./ sum(Ws)
end

convert_HMS(s) = string(floor(Int, s÷3600), ":", lpad(floor(Int, s%3600÷60), 2, '0'), ":", lpad(floor(Int, s%60), 2, '0'));

function get_recycled_pvec(all_particles, logprior_funcs, targetinfos, thres)
    iter = length(all_particles) - 1
    logws = [fill(NaN, length(ps)) for ps in all_particles];
    cached_dtors = [fill(NaN, length(ps)) for ps in all_particles];
    logZs = fill(NaN, iter+1); logZs[1] = 0.;
    pop_sizes = length.(all_particles);

    logpriors_by_iter = logprior_funcs[first.(targetinfos)];

    for i1 in 2:(iter+2)
        _, i1_β = targetinfos[min(i1, iter+1)]
        for i2 in 1:(i1-1)
            states = getproperty.(all_particles[i2], :state)
            loglikes = getproperty.(all_particles[i2], :loglike)
            ntors = logpriors_by_iter[min(i1, iter+1)].(states) .+ i1_β .* loglikes
            if i2 == i1 - 1
                cached_dtors[i2] .= [
                    logsumexp([
                        logpriors_by_iter[s](state) + targetinfos[s][2]*loglike - logZs[s] + log(pop_sizes[s]) for s in 1:(i1-1)
                    ]) for (state, loglike) in zip(states, loglikes)
                ]
            else
                new_terms = logpriors_by_iter[i1-1].(states) .+ targetinfos[i1-1][2].*loglikes .- logZs[i1-1] .+ log(pop_sizes[i1-1])
                cached_dtors[i2] .= logaddexp.(cached_dtors[i2], new_terms)
            end
            dtors = cached_dtors[i2] .- log(sum(pop_sizes[1:i1-1]))
            logws[i2] = map(x -> isfinite(x) ? x : -Inf, ntors .- dtors)        
        end
        if i1 <= iter + 1
            cat_logws = [logw for i2 in 1:(i1-1) for logw in logws[i2]]
            logZs[i1] = logsumexp(cat_logws) - log(length(cat_logws))
        end
    end

    cat_logws = reduce(vcat, logws);
    cat_Ws = exp.(cat_logws .- logsumexp(cat_logws));
    return get_pvec(reduce(vcat, all_particles), thres, cat_Ws)    
end

function get_pvec(particles, thres, Ws=ones(length(particles)))
    return get_mode_probs(getproperty.(particles, :state), thres, Ws)
end

logZs_fname = joinpath(@__DIR__, "output/logZs.jld2");
@load logZs_fname rAMIS_logZvecs BS_logZvecs all_times;
pvecs_BS = [exp.(logZvec .- logsumexp(logZvec)) for logZvec in BS_logZvecs];
pvecs_rAMIS = [exp.(logZvec .- logsumexp(logZvec)) for logZvec in rAMIS_logZvecs];
tvds_rAMIS = [0.5sum(abs, pvec_BS .- pvec_rAMIS) for (pvec_BS, pvec_rAMIS) in zip(pvecs_BS, pvecs_rAMIS)];

# run_str, n_priors, temper_prior = ("SMC362", 20, true)
# run_str, n_priors, temper_prior = ("SMC062", 36, true)

run_str, n_priors, temper_prior = ("SMC662", 1, true)
# run_str, n_priors, temper_prior = ("SMC602", 1, true)
# run_str, n_priors, temper_prior = ("SMC660", 1, true)
run_str, n_priors, temper_prior = ("SMC262", 1, false)

SMC_fname = joinpath(@__DIR__, "output/$(run_str).jld2");

(μ0, σ0) = (n_priors == 1) ? (-8., 4.) : (-4., 3.)
slab_seq = interpolate_ss.(0:(n_priors-1), μ0, σ0, μ_slab, σ_slab, temper_prior);
spike_seq = interpolate_ss.(0:(n_priors-1), μ0, σ0, μ_spike, σ_spike, temper_prior);
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
thres = thres_vec[end];
pvecs_SMC = [Float64[] for _ in 1:n_feasible];
pvecs_rSMC = [Float64[] for _ in 1:n_feasible];
hours_SMC = [0. for _ in 1:n_feasible];
ns_nuts = [0 for _ in 1:n_feasible];
targetinfos_vec = [Tuple{Int,Float64}[] for _ in 1:n_feasible];

if isfile(SMC_fname)
    @load SMC_fname pvecs_SMC pvecs_rSMC hours_SMC ns_nuts targetinfos_vec;
end;

tmp = 0;
@showprogress for dir_idx in 1:n_feasible
    OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
    fname = "$OUTDIR/$(run_str).jld2"
    if hours_SMC[dir_idx] > 0
        tmp += 1
        continue
    end
    if isfile(fname)
        # @load fname all_particles ess_vec iter targetinfos smc_times;
        @load fname all_particles iter targetinfos smc_times;
        # println(targetinfos[end])
        if targetinfos[end] == (n_priors, 1.)
            tmp += 1
            pvecs_SMC[dir_idx] = get_pvec(all_particles[end], thres)
            pvecs_rSMC[dir_idx] = get_recycled_pvec(all_particles, logprior_funcs, targetinfos, thres)
            hours_SMC[dir_idx] = sum(smc_times)/3600
            ns_nuts[dir_idx] = sum(ps[1].n_nuts for ps in all_particles[2:end])
            targetinfos_vec[dir_idx] = targetinfos
        else
            @info dir_idx iter targetinfos[end] 
        end
    end
end
tmp

findall(isempty.(pvecs_SMC))

tvds_SMC = [dir_idx => 0.5sum(abs, pvec_BS .- pvec_SMC) for (dir_idx, pvec_BS, pvec_SMC) in zip(1:n_feasible, pvecs_BS, pvecs_SMC) if !isempty(pvec_SMC)];
summarystats(last.(tvds_SMC)) |> display

tvds_rSMC = [dir_idx => 0.5sum(abs, pvec_BS .- pvec_rSMC) for (dir_idx, pvec_BS, pvec_rSMC) in zip(1:n_feasible, pvecs_BS, pvecs_rSMC) if !isempty(pvec_rSMC)];
summarystats(last.(tvds_rSMC)) |> display

summarystats(filter(!iszero, ns_nuts)) |> display
summarystats(filter(!iszero, hours_SMC)) |> display
summarystats(filter(!iszero, length.(targetinfos_vec)) .- 1) |> display

@save SMC_fname pvecs_SMC pvecs_rSMC hours_SMC ns_nuts targetinfos_vec;

summarystats((ns_nuts ./ (length.(targetinfos_vec) .- 1)) .|> mean)
summarystats(375 ./ hours_SMC)

hist(filter(!iszero, ns_nuts), axis=(xlabel="Number of NUTS iterations",))
hist(filter(!iszero, hours_SMC), axis=(xlabel="Computational time (hours)",))
hist(filter(!iszero, length.(targetinfos_vec)) .- 1, axis=(xlabel="Number of SMC iterations", xticks=1:100))


## Ad hoc correaction for data21_model50
begin
    dir_idx = 21
    pvecs_BSnew = copy.(pvecs_BS)
    BS_logZvec = copy(BS_logZvecs[dir_idx])
    BS_logZvec[50] = 70.33643911435237
    pvecs_BSnew[dir_idx] = exp.(BS_logZvec .- logsumexp(BS_logZvec))
end;

tvds_SMC = [dir_idx => 0.5sum(abs, pvec_BS .- pvec_SMC) for (dir_idx, pvec_BS, pvec_SMC) in zip(1:n_feasible, pvecs_BSnew, pvecs_SMC) if !isempty(pvec_SMC)];
summarystats(last.(tvds_SMC)) |> display

tvds_rSMC = [dir_idx => 0.5sum(abs, pvec_BS .- pvec_rSMC) for (dir_idx, pvec_BS, pvec_rSMC) in zip(1:n_feasible, pvecs_BSnew, pvecs_rSMC) if !isempty(pvec_rSMC)];
summarystats(last.(tvds_rSMC)) |> display


## Plots of TVD and pvec
using SpecialFunctions
mad_func(N, p) = exp(floor(N*p)*log(p)+(N-floor(N*p))*log(1-p)+logabsbinomial(N-1, floor(Int, N*p))[1])

# etvds = map(pvec -> sum(sqrt, pvec)/sqrt(2π*10^4), pvecs_BS);
etvds = map(pvec -> sum(p->p*mad_func(5000,p), pvec), pvecs_BS);
begin
    f = Figure()
    ax = Axis(
        f[1,1], limits=((0., 1.025*maximum(etvds)), (0., 1.025*maximum(tvds_SMC .|> last))),
        xlabel="Expected TVD assuming ideal sampling", ylabel="Actual TVD (SMC)"
    )
    lines!(
        [0, maximum(tvds_SMC .|> last)], [0, maximum(tvds_SMC .|> last)], 
        color=:grey30, alpha=0.7, linestyle=:dash
    )
    plot_idxs = first.(tvds_SMC)
    sc = scatter!(etvds[plot_idxs], last.(tvds_SMC), alpha=0.7, color=ns_nuts[plot_idxs])
    Colorbar(f[1,2], sc, label="Number of NUTS iterations")
    display(f)
    save(joinpath(@__DIR__, "imgs/$(run_str)_TVDs.png"), f, px_per_unit=4)
end

begin
    f = Figure()
    ax = Axis(
        f[1,1], limits=((0., 1.025*maximum(etvds)), (0., 1.025*maximum(tvds_rSMC .|> last))),
        xlabel="Expected TVD assuming ideal sampling", ylabel="Actual TVD (recylced SMC)"
    )
    lines!(
        [0, maximum(tvds_rSMC .|> last)], [0, maximum(tvds_rSMC .|> last)], 
        color=:grey30, alpha=0.7, linestyle=:dash
    )
    plot_idxs = first.(tvds_rSMC)
    sc = scatter!(etvds[plot_idxs], last.(tvds_rSMC), alpha=0.7, color=ns_nuts[plot_idxs])
    Colorbar(f[1,2], sc, label="Number of NUTS iterations")
    display(f)
    save(joinpath(@__DIR__, "imgs/$(run_str)_TVDs_recycled.png"), f, px_per_unit=4)
end

sort(tvds_SMC, by=last)

hist(last.(tvds_SMC), axis=(xlabel="TVD relative to bridge sampling",), bins=0:0.005:(0.005+maximum(last.(tvds_SMC))))

begin
    # Representative case looks ok
    # dir_idx = n_feasible

    # ?
    # dir_idx = 20

    # AMIS and SMC agree
    # dir_idx = 21

    # AMIS and bridge agree
    dir_idx = 39

    # AMIS and bridge agree
    # dir_idx = 3

    # AMIS and bridge agree
    # dir_idx = 31 

    n_plot = 20
    plot_order = sortperm(pvecs_BS[dir_idx], rev=true)[1:n_plot]

    COLORS = Makie.wong_colors()[[4,2,5]]

    f = Figure(size=(400, 640))
    ax = Axis(
        f[1,1], yreversed=true,
        ylabel="Models ranked by bridge sampling", xlabel="Model posterior probability",
        limits=((0, nothing), (0.3, n_plot+0.7)), yticks = (1:n_plot, string.(plot_order))
    )
    width = 0.25
    barplot!(
        (1:n_plot) .- width, pvecs_rAMIS[dir_idx][plot_order], direction=:x,
        gap=0, width=width, color=COLORS[1], label="Robust AMIS"
    )
    barplot!(
        (1:n_plot) , pvecs_BS[dir_idx][plot_order], direction=:x,
        gap=0, width=width, color=COLORS[2], label="Bridge sampling"
    )
    barplot!(
        (1:n_plot) .+ width, pvecs_SMC[dir_idx][plot_order], direction=:x,
        gap=0, width=width, color=COLORS[3], label="Spike-and-slab SMC"
    )
    axislegend(ax, position=:rb)
    display(f)
end

## Plot model posterior probability estimates with identifiability as colour
vrats_fname = joinpath(@__DIR__, "output/var_ratios.jld2");
@load vrats_fname var_ratios;

begin
    f = Figure()
    ax = Axis(
        f[1,1], aspect=DataAspect(), xticks=0:0.2:1,
        xlabel="SMC", xlabelsize=18, 
        xticklabelsize=16, yticklabelsize=16,
        ylabel="Bridge sampling", ylabelsize=18
    )
    vcat_pvecs_SMC = reduce(vcat, pvecs_SMC)
    pmax = pvecs_BS .|> maximum |> maximum
    lines!([0, pmax], [0, pmax], color=(:grey10, 0.8), linestyle=:dash)
    sc = scatter!(
        vcat_pvecs_SMC, reduce(vcat, pvecs_BS),
        color=sqrt.(vec(var_ratios')), colorrange=(0, 0.6), #colormap=Reverse(:viridis),
        highclip=:yellow,
        alpha=0.6, markersize=7
    )
    Colorbar(
        f[1,2], sc, ticklabelsize=16, alignmode=Mixed(right=0),
        label="Max posterior-to-prior SD ratio", labelsize=18
    )
    display(f)
    save(joinpath(@__DIR__, "imgs/SMC_10000_modelprobs.png"), f, px_per_unit=4)
end

# keep = findall(p -> all(p.state .> -8), all_particles[end])
# hist([p.logtarget for p in all_particles[end]][keep] .+ 6log(2), bins=65:0.5:85, alpha=0.5, normalization=:pdf);
# hist!(vec(chn[:lp].data), bins=65:0.5:85, alpha=0.5, normalization=:pdf);
# display(current_figure())

