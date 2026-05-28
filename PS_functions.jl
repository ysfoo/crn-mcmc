include(joinpath(@__DIR__, "stats_helpers.jl"));
include(joinpath(@__DIR__, "plot_helpers.jl"));

using Distributions, Random, Statistics, StatsBase
using Distances, LinearAlgebra, LogExpFunctions, NearestNeighbors
using AdvancedHMC, Bijectors, LogDensityProblems, LogDensityProblemsAD, ForwardDiff, PreallocationTools
using Accessors, JLD2, ProgressMeter
using VideoIO


struct PSParticle
    state::AbstractVector{Float64}
    loglike::Float64   # log likelihood
    logtarget::Float64 # log target density
    stepsize::Float64  # NUTS step size
    n_nuts::Int64      # number of NUTS iterations
    info::NamedTuple
end


# Runs one PS iteration: reweight → resample → move.
# Modifies all_particles, all_dtors, logZs, targetinfos.
# Returns (all_logws, npass, fig, smc_time).
function PS_iteration!(
    iter, all_particles, all_dtors, logZs,
    targetinfos, pop_size, target_ess, 
    logprior_funcs, ldp_builder, move_func, rerun_func, rngs; 
    max_npass=10, parallel=false, pbar_lines=pop_size, vid_path=nothing, make_fig=nothing, verbose=0
)
    t_start = time()

    # Compute densities of all samples under each target so far, weighted by population size
    pop_sizes = length.(all_particles)
    prev_prior_idx, prev_β = prev_targetinfo = targetinfos[end]
    prev_logprior_func = logprior_funcs[prev_prior_idx]
    prev_logZ = logZs[end]
    prev_pop_size = pop_sizes[end]
    for t in 1:iter
        states = getproperty.(all_particles[t], :state)
        loglikes = getproperty.(all_particles[t], :loglike)
        if t == iter # evaluate latest population for all targets
            push!(all_dtors, [
                logsumexp([
                    begin
                        prior_idx_s, β_s = targetinfos[s]
                        logprior_funcs[prior_idx_s](state) + β_s*loglike - logZs[s] + log(pop_sizes[s])
                    end for s in 1:iter
                ]) for (state, loglike) in zip(states, loglikes)
            ])
        else # evaluate earlier populations for latest target
            new_terms = prev_logprior_func.(states) .+ prev_β .* loglikes .- prev_logZ .+ log(prev_pop_size)
            all_dtors[t] .= logaddexp.(all_dtors[t], new_terms)
        end
    end    
    
    target, targetinfo, all_logws = build_target(iter, all_particles, all_dtors, prev_targetinfo, target_ess, logprior_funcs, ldp_builder; verbose)
    
    # Reweight
    cat_logws = reduce(vcat, all_logws)
    @assert !any(isnan, cat_logws)
    if isnothing(target)
        return (all_logws, 0, nothing, 0.0) # (all_logws, npass, fig, smc_time)
    end
    push!(logZs, logsumexp(cat_logws) - log(length(cat_logws)))
    rng = rngs[1]
    prior_idx, β = targetinfo
    push!(targetinfos, targetinfo)

    # Resample
    cat_ws = exp.(cat_logws .- maximum(cat_logws))
    sampled_idxs = stratified_sampling(cat_ws, pop_size; rng=rng)
    states_before_move = getproperty.(reduce(vcat, all_particles)[sampled_idxs], :state)

    # Lookup particle-specific step size and acceptance rate, and reset number of NUTS iterations
    prev_btree = BallTree(reduce(hcat, p.state for p in all_particles[end]))
    lookup, _ = nn(prev_btree, reduce(hcat, states_before_move))
    particles = [
        PSParticle(state, NaN, NaN, p_lookup.stepsize, 0, p_lookup.info)
        for (state, p_lookup) in zip(states_before_move, all_particles[end][lookup])
    ]

    # Move
    npass = 0
    # stats_before = Base.gc_num()
    smc_time = time() - t_start
    while npass < max_npass
        npass += 1
        pbar = Progress(pop_size; desc="Iter $iter, pass $(npass)")
        if parallel
            counter = Threads.Atomic{Int}(0)
            targets = [customcopy(target) for _ in 1:pop_size]
            thread_times = zeros(Threads.nthreads())
            GC.gc(false)            

            Threads.@threads for i in 1:pop_size
                particle = particles[i]
                thread_times[Threads.threadid()] += @elapsed particles[i] = move_func(rngs[Threads.threadid()], particle, targets[i])
                prev_done = Threads.atomic_add!(counter, 1)
                if (prev_done + 1) % (pop_size ÷ pbar_lines) == 0
                    ProgressMeter.update!(pbar, prev_done + 1)
                end
            end
            if verbose > 0
                display(thread_times)
            end
            smc_time += sum(thread_times)
        else
            for i in 1:pop_size
                particle = particles[i]
                smc_time += @elapsed particles[i] = move_func(rng, particle, target)
                if i % (pop_size ÷ pbar_lines) == 0
                    ProgressMeter.update!(pbar, i)
                end
            end
        end
        ProgressMeter.finish!(pbar)

        rerun_func(particles, states_before_move, iter, targetinfo; verbose) || break

        flush(stdout)
        flush(stderr)
    end

    logprior_func = logprior_funcs[prior_idx]
    particles = [@set p.loglike = (p.logtarget - logprior_func(p.state)) / β for p in particles]
    push!(all_particles, particles)

    agg_acc_rate = mean(p.info.agg_acc_rate for p in particles)
    median_esjd = median([p.info.esjd for p in particles])

    if verbose > 0
        # stats_after = Base.gc_num()
        # gc_time = (stats_after.total_time - stats_before.total_time) / 1e9  # seconds
        # mark_time = (stats_after.total_mark_time - stats_before.total_mark_time) / 1e9  # seconds
        # sweep_time = (stats_after.total_sweep_time - stats_before.total_sweep_time) / 1e9  # seconds
        # n_pauses = stats_after.pause - stats_before.pause
        # @info "GC" gc_time mark_time sweep_time n_pauses  
        @info "Iter $iter post-MCMC" agg_acc_rate median_esjd logZs[end]
        display(summarystats(getproperty.(particles, :stepsize)))
        flush(stdout)
        flush(stderr)
    end    

    fig = nothing
    if !isnothing(vid_path)
        fig = make_fig(particles, iter)
        save(joinpath(vid_path, "iter$(iter).png"), fig, px_per_unit=4)
    end

    return all_logws, npass, fig, smc_time # (all_logws, npass, fig, smc_time)
end


function build_target(iter, all_particles, all_dtors, prev_targetinfo, target_ess, logprior_funcs, ldp_builder; verbose=0, min_temp=1e-8)
    n_priors = length(logprior_funcs)
    tot_pop_size = sum(length, all_particles)
    prev_prior_idx, prev_β = prev_targetinfo

    @assert length(all_dtors) == length(all_particles)
    @assert iter > 0    

    if prev_prior_idx == n_priors && prev_β >= 1 - min_temp
        logprior_func = logprior_funcs[prev_prior_idx]
        all_logws = [
            begin
                ntors = logprior_func.(getproperty.(particles, :state)) .+ prev_β .* getproperty.(particles, :loglike)
                tmp_logws = ntors .- (dtors .- log(tot_pop_size))
                tmp_logws[.!isfinite.(tmp_logws)] .= -Inf
                tmp_logws
            end for (dtors, particles) in zip(all_dtors, all_particles)
        ]
        return (nothing, nothing, all_logws)
    end
    
    # Prior tempering
    curr_prior_idx = min(n_priors, prev_β > 0.01 ? prev_prior_idx + 1 : prev_prior_idx)
    logprior_func = logprior_funcs[curr_prior_idx]

    # Likelihood tempering
    cat_logpriors = logprior_func.([p.state for ps in all_particles for p in ps])
    cat_loglikes = [p.loglike for ps in all_particles for p in ps]
    cat_dtors = reduce(vcat, all_dtors) .- log(tot_pop_size)

    prev_ess = compute_ess(cat_logpriors .+ prev_β .* cat_loglikes .- cat_dtors)
    if !isfinite(prev_ess)
        display(extrema(cat_logpriors))
        display(extrema(cat_loglikes))
        display(extrema(cat_dtors))
        @assert isfinite(prev_ess)
    end

    if (prev_ess < target_ess) && (curr_prior_idx > prev_prior_idx)
        curr_prior_idx -= 1
        logprior_func = logprior_funcs[curr_prior_idx]
        cat_logpriors = logprior_func.([p.state for ps in all_particles for p in ps])
        cat_loglikes = [p.loglike for ps in all_particles for p in ps]
        cat_dtors = reduce(vcat, all_dtors) .- log(tot_pop_size)
        prev_ess = compute_ess(cat_logpriors .+ cat_loglikes .* prev_β .- cat_dtors)
    end
    curr_β = bisection_search(
        target_ess, 
        (β) -> compute_ess(cat_logpriors .+ cat_loglikes .* β .- cat_dtors), 
        max(min_temp, prev_β), 1.; tol=min_temp, is_increasing=false
    )

    if verbose > 0
        @info "Iter $iter tempering" curr_prior_idx curr_β prev_ess
    end

    all_logws = [
        begin
            ntors = logprior_func.(getproperty.(particles, :state)) .+ curr_β .* getproperty.(particles, :loglike)
            tmp_logws = ntors .- (dtors .- log(tot_pop_size))
            tmp_logws[.!isfinite.(tmp_logws)] .= -Inf
            tmp_logws
        end for (dtors, particles) in zip(all_dtors, all_particles)
    ]

    return (ldp_builder(logprior_func, curr_β), (curr_prior_idx, curr_β), all_logws)
end

function run_PS(
    pop_size, target_ess, init_sampler, logprior_funcs, ldp_builder, move_func, rerun_func, fname;
    init_pop_size=pop_size, init_stepsize::Float64=0.01, 
    max_npass=10, verbose=0, vid_path=nothing, make_fig=nothing,
    parallel=false, pbar_lines=pop_size,
)
    rngs = parallel ? [Xoshiro() for _ in 1:Threads.nthreads()] : [Random.default_rng()]
    rng = rngs[1]
    initstates = [init_sampler(rng) for _ in 1:init_pop_size] 

    iter = 0
    init_logprior_func = logprior_funcs[1]
    logpriors = init_logprior_func.(initstates)
    tmp_target = ldp_builder(init_logprior_func, 1.) # use β = 1 to extract likelihood
    all_particles = [PSParticle.(
        initstates,
        LogDensityProblems.logdensity.(Ref(tmp_target), initstates) .- logpriors,
        logpriors, # actual logtarget has β = 0
        init_stepsize, 0, Ref(NamedTuple())
    )]
    targetinfos = [(1, 0.)]
    all_dtors = Vector{Float64}[]
    all_logws = Vector{Float64}[]
    logZs = [0.]
    npass_vec, smc_times = Int64[], Float64[]

    if !isnothing(vid_path)
        figs = Figure[]
        mkpath(vid_path)
    end

    while true
        iter += 1
        all_logws, npass, fig, smc_time = PS_iteration!(
            iter, all_particles, all_dtors, logZs,
            targetinfos, pop_size, target_ess, 
            logprior_funcs, ldp_builder, move_func, rerun_func, rngs;
            max_npass, parallel, pbar_lines, vid_path, make_fig, verbose
        ) # (all_logws, npass, fig, smc_time)

        if npass == 0
            iter -= 1
            @save fname all_particles all_dtors all_logws logZs iter targetinfos npass_vec smc_times
            break
        end

        !isnothing(vid_path) && push!(figs, fig)
        push!(npass_vec, npass)
        push!(smc_times, smc_time)

        @save fname all_particles all_dtors all_logws logZs iter targetinfos npass_vec smc_times
    end

    if !isnothing(vid_path)
        VideoIO.save(
            joinpath(vid_path, "iters.mp4"),
            [CairoMakie.Colors.RGB.(colorbuffer(fig)) for fig in figs],
            framerate=3, encoder_options=(crf=23, preset="medium")
        )
    end
end


# Resumes a run_PS that was interrupted, reading state from `fname``.
function resume_PS(
    pop_size, target_ess, logprior_funcs, ldp_builder, move_func, rerun_func, fname;
    max_npass=10, verbose=0, vid_path=nothing, make_fig=nothing, parallel=false, pbar_lines=pop_size
)
    @load fname all_particles all_dtors all_logws logZs iter targetinfos npass_vec smc_times
    @assert length(all_particles) == (iter + 1)
    rngs = parallel ? [Xoshiro() for _ in 1:Threads.nthreads()] : [Random.default_rng()]

    if !isnothing(vid_path)
        mkpath(vid_path)
        figs = make_fig.(all_particles[2:end], 1:iter)   
    end

    @info "Resuming PS from iter $iter"

    while true
        iter += 1
        all_logws, npass, fig, smc_time = PS_iteration!(
            iter, all_particles, all_dtors, logZs,
            targetinfos, pop_size, target_ess, 
            logprior_funcs, ldp_builder, move_func, rerun_func, rngs;
            max_npass, parallel, pbar_lines, vid_path, make_fig, verbose
        ) # (all_logws, npass, fig, smc_time)

        if npass == 0
            iter -= 1
            @save fname all_particles all_dtors all_logws logZs iter targetinfos npass_vec smc_times
            break
        end

        !isnothing(vid_path) && push!(figs, fig)
        push!(npass_vec, npass)
        push!(smc_times, smc_time)

        @save fname all_particles all_dtors all_logws logZs iter targetinfos npass_vec smc_times
    end

    if !isnothing(vid_path)
        VideoIO.save(
            joinpath(vid_path, "iters.mp4"),
            [CairoMakie.Colors.RGB.(colorbuffer(fig)) for fig in figs],
            framerate=3, encoder_options=(crf=23, preset="medium")
        )
    end
end


function nuts_move(rng, particle, target; n_nuts=5)
    metric = DiagEuclideanMetric(LogDensityProblems.dimension(target))
    h = Hamiltonian(metric, target, ForwardDiff)

    stepsize = particle.stepsize
    # if particle.n_nuts == 0 && hasproperty(particle.info, :avg_acc_rate)
    #     stepsize *= 1.5^((particle.info.avg_acc_rate-0.8)/0.2) # adapt only at start of PS move phase, not each NUTS
    # end
    integrator = Leapfrog(stepsize)
    κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

    θs, stats = sample(rng, h, κ, particle.state, n_nuts, NoAdaptation(), 0; verbose=false)

    if particle.n_nuts == 0
        sum_sqjdist = 0.
        sum_acc_rate = 0.
        delta = θs[end] .- particle.state
    else
        sum_sqjdist = particle.info.esjd * particle.n_nuts
        sum_acc_rate = particle.info.agg_acc_rate * particle.n_nuts
        delta = particle.info.delta .+ θs[end] .- particle.state
    end

    agg_n_nuts = particle.n_nuts + n_nuts
    acc_rate_incr = sum(map(s -> s.acceptance_rate, stats))
    sum_acc_rate += acc_rate_incr
    sum_sqjdist += sum(abs2, θs[1] .- particle.state)
    for i in 2:n_nuts
        sum_sqjdist += sum(abs2, θs[i] .- θs[i-1])
    end

    info = (
        curr_acc_rate = acc_rate_incr / n_nuts,
        agg_acc_rate = sum_acc_rate / agg_n_nuts,
        esjd = sum_sqjdist / agg_n_nuts,
        delta = delta
    )

    # adapt stepsize after NUTS using most recent acceptance rate
    stepsize *= 1.5^((info.agg_acc_rate-0.8)/0.2)

    return PSParticle(θs[end], NaN, stats[end].log_density, stepsize, agg_n_nuts, info)

    # agg_n_nuts = particle.n_nuts + n_nuts
    # sum_acc_rate += sum(map(s -> s.acceptance_rate, stats))
    # sum_sqjdist += sum(abs2, θs[1] .- particle.state)
    # for i in 2:n_nuts
    #     sum_sqjdist += sum(abs2, θs[i] .- θs[i-1])
    # end    

    # info = (
    #     avg_acc_rate = sum_acc_rate / agg_n_nuts,
    #     esjd = sum_sqjdist / agg_n_nuts,
    #     delta = delta
    # )

    # return PSParticle(θs[end], NaN, stats[end].log_density, stepsize, agg_n_nuts, info)
end
