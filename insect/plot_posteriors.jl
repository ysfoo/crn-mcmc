include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../plot_helpers.jl"));

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, PEtab, Random
using JLD2, ProgressMeter, Suppressor
using Bijectors, LogDensityProblems, LogDensityProblemsAD
using AdvancedHMC, Bijectors, BridgeSampling, LogDensityProblems, LogDensityProblemsAD, MCMCChains, PSIS, Turing


# Posterior plots

dir_idx = 2;
genmodel_idx = feasible_idxs[dir_idx];

param_labels = [
    L"\lambda_1", L"\lambda_2", L"\rho", 
    L"\delta_1", L"\delta_2", L"\delta_3", 
    L"\kappa_1", L"\kappa_2", L"K_3", L"\sigma"
];
sym2label = Dict(zip(Symbol.(parameters(models[end])), param_labels))

@showprogress for model_idx in 1:n_models     
    d = nparams[model_idx]
    ps = parameters(models[model_idx])

    INFDIR = joinpath("/scratch/punim0638/ysfoo/crn-mcmc/insect/output/data$(dir_idx)") # inference result directory

    mcmc_fname = joinpath(INFDIR, "gm_MCMC_model$(model_idx).jld2")
    @nowarn_load mcmc_fname chn
    trace = chn.value[:,1:d,1].data;
    X = Matrix(trace');

    f = plot_pairs(
        # eachcol(exp10.(X)),
        eachcol(X),
        # title="Posterior\nsamples under model $model_idx for data generated from model $genmodel_idx",
        figsize=(120*d+180, 120*d),
        scatter_kwargs=(color=(:grey, 0.2), markersize=5),
    ); 

    for (i1, p1) in enumerate(ps)
        for (i2, p2) in enumerate(ps)
            idx = (i2-1)*d + i1
            ax = f.content[idx]
            if i1 == i2
                # autolimits!(ax)
                # ax_limits = ax.finallimits[]
                # xs = -10:0.01:10
                # dist = i1 == d ? Normal(-1, 1) : Normal(0, 2)
                # lines!(ax, xs, pdf.(Ref(dist), xs), color=:black)
                # limits!(ax, ax_limits)
                # xlims!(ax, (ax_limits.origin[1], ax_limits.origin[1]+ax_limits.widths[1]))
            end
            if i2 ∈ [1, d]
                ax.yaxisposition = i2 == 1 ? :left : :right
                ax.ylabel = (i1 == i2) ? "" : L"\log_{10} %$(sym2label[Symbol(p1)])"
                # ax.ylabel = (i1 == i2) ? "" : L"%$(sym2label[Symbol(p1)])"
                ax.ylabelsize = 18
                ax.yticklabelsize = 12
                ax.yticksvisible = i1 != i2
                ax.yticklabelsvisible = i1 != i2
            else
                ax.yticksvisible = false
                ax.yticklabelsvisible = false
            end
            
            if i1 ∈ [1, d]
                ax.xaxisposition = i1 == 1 ? :top : :bottom
                ax.xlabel = L"\log_{10} %$(sym2label[Symbol(p2)])"
                # ax.xlabel = L"%$(sym2label[Symbol(p2)])"
                ax.xlabelsize = 18
                ax.xticklabelrotation = π/4
                ax.xticklabelsize = 12
            else
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
            end
        end
    end

    Legend(
        f[:,d+1], 
        [
            MarkerElement(color = :grey, marker=:circle, markersize=8), 
            PolyElement(color=Makie.wong_colors()[1], strokewidth = 0, points = Point2f[(0, 0.25), (1, 0.25), (1, 0.75), (0, 0.75)])
        ],
        ["Posterior\nsamples", "Log posterior\ndensity"],
        labelsize=18, rowgap=10,        
    )
    
    save_dir = mkpath(joinpath(@__DIR__, "imgs/data$(dir_idx)_logparams/"))
    mkpath(save_dir)
    save("$(save_dir)/model$(model_idx).png", f)

    ## Now with prior
    f = plot_pairs(
        # eachcol(exp10.(X)),
        eachcol(X),
        [[i == d ? -1 : 0 for i in 1:d]],
        [diagm([i == d ? 1 : 4 for i in 1:d])],
        # title="Posterior\nsamples under model $model_idx for data generated from model $genmodel_idx",
        figsize=(120*d+180, 120*d),
        scatter_kwargs=(color=(:grey, 0.2), markersize=5),
        ellipse_kwargs=(color=:black,),
        hist_kwargs=(fillto=1e-4,),
    ); 

    for (i1, p1) in enumerate(ps)
        for (i2, p2) in enumerate(ps)
            idx = (i2-1)*d + i1
            ax = f.content[idx]
            if i1 == i2
                i = i1
                xs = (i == d) ? (-4.5:0.05:2.5) : (-7:0.05:7)
                limits!(ax, extrema(xs), (2e-4, 1))
                lines!(ax, xs, pdf.(Ref(i == d ? Normal(-1, 1) : Normal(0, 2)), xs), color=:black)
            else
                xs = (i2 == d) ? (-4.5:0.05:2.5) : (-7:0.05:7)
                ys = (i1 == d) ? (-4.5:0.05:2.5) : (-7:0.05:7)
                limits!(ax, extrema(xs), extrema(ys))
            end
            if i2 ∈ [1, d]
                ax.yaxisposition = i2 == 1 ? :left : :right
                ax.ylabel = (i1 == i2) ? "" : L"\log_{10} %$(sym2label[Symbol(p1)])"
                # ax.ylabel = (i1 == i2) ? "" : L"%$(sym2label[Symbol(p1)])"
                ax.ylabelsize = 18
                ax.yticklabelsize = 12
                ax.yticksvisible = i1 != i2
                ax.yticklabelsvisible = i1 != i2
            else
                ax.yticksvisible = false
                ax.yticklabelsvisible = false
            end
            
            if i1 ∈ [1, d]
                ax.xaxisposition = i1 == 1 ? :top : :bottom
                ax.xlabel = L"\log_{10} %$(sym2label[Symbol(p2)])"
                # ax.xlabel = L"%$(sym2label[Symbol(p2)])"
                ax.xlabelsize = 18
                ax.xticklabelrotation = π/4
                ax.xticklabelsize = 12
            else
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
            end
        end
    end

    Legend(
        f[:,d+1], 
        [
            MarkerElement(color = :grey, marker=:circle, markersize=8), 
            PolyElement(color=Makie.wong_colors()[1], strokewidth = 0, points = Point2f[(0, 0.25), (1, 0.25), (1, 0.75), (0, 0.75)]),
            LineElement(color=:black, )
        ],
        ["Posterior\nsamples", "Log posterior\ndensity", "Log prior\ndensity"],
        labelsize=18, rowgap=10,
    )
    
    save_dir = mkpath(joinpath(@__DIR__, "imgs/data$(dir_idx)_logparams_prior/"))
    mkpath(save_dir)
    save("$(save_dir)/model$(model_idx).png", f)
end