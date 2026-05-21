include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../gaussian_mixtures.jl"));
include(joinpath(@__DIR__, "../plot_helpers.jl"));

using Turing, MCMCChains, ProgressMeter
using PDMats, LogExpFunctions
using BridgeSampling, PSIS, StatsBase
using KernelDensity

OUTDIR = joinpath(@__DIR__, "output")
@load "$OUTDIR/MAPs.jld2" model_fits;

n_seed = 100;

# MCMC diagnostics
miness_dict = Dict{Symbol,Vector{Float64}}();
maxrhat_dict = Dict{Symbol,Vector{Float64}}();
duration_dict = Dict{Symbol,Vector{Float64}}();
vratio_dict = Dict{Symbol,Vector{Float64}}();
for model_sym in model_syms
    miness_vec = Float64[]
    maxrhat_vec = Float64[]
    duration_vec = Float64[]
    vratio_vec = Float64[]
    priorvars = getproperty.(priors_dict[model_sym], :σ) .^ 2
    d = length(priorvars)
    @showprogress for s in 1:n_seed
        INFDIR = joinpath(@__DIR__, "output/seed$s");
        fname = joinpath(INFDIR, "chains_$(model_sym).jld2");
        @load fname chn ess_df
        trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3]);
        X = reshape(trace, d, :);
        postvars = vec(var(X; dims=2))

        push!(miness_vec, minimum(ess_df.nt.ess))
        push!(maxrhat_vec, maximum(rhat(chn).nt.rhat))
        push!(duration_vec, MCMCChains.compute_duration(chn))
        push!(vratio_vec, maximum(postvars ./ priorvars))
    end
    miness_dict[model_sym] = miness_vec
    maxrhat_dict[model_sym] = maxrhat_vec
    duration_dict[model_sym] = duration_vec
    vratio_dict[model_sym] = vratio_vec
end

[display(summarystats(miness_dict[sym])) for sym in model_syms];
[display(summarystats(maxrhat_dict[sym])) for sym in model_syms];
[display(summarystats(duration_dict[sym])) for sym in model_syms];
[display(summarystats(vratio_dict[sym] .|> sqrt)) for sym in model_syms];

Zhat_BIC_dict = Dict(
    sym => begin
        MAP = model_fits[sym].value.MAP
        # hess = model_fits[sym].value.hess
        # Σ = inv(PDMat(hermitianpart!(hess)))
        target = target_dict[sym]
        prior_dists = priors_dict[sym]
        ML = LogDensityProblems.logdensity(target, MAP) - compute_logprior(prior_dists, MAP)
        ML - 0.5*log(n_times)*nparam_dict[sym]
    end for sym in model_syms
)

Zhat_BS_dict = Dict{Symbol,Vector{Float64}}();
re2_BS_dict = Dict{Symbol,Vector{Float64}}();
for model_sym in model_syms
    resvec = [
        begin
            INFDIR = joinpath(@__DIR__, "output/seed$s");
            fname = joinpath(INFDIR, "BS_$(model_sym).jld2");
            @load fname timed_res
            timed_res.value
        end for s in 1:n_seed
    ]
    Zhat_BS_dict[model_sym] = getproperty.(resvec, :value)
    re2_BS_dict[model_sym] = getproperty.(error_estimate.(resvec), :rmse)
end

Zhat_LIS_dict = Dict{Symbol,Vector{Float64}}();
ess_LIS_dict = Dict{Symbol,Vector{Float64}}();
for model_sym in model_syms
    resvec = [
        begin
            INFDIR = joinpath(@__DIR__, "output/seed$s");
            fname = joinpath(INFDIR, "laplace_IS_$(model_sym).jld2");
            @load fname timed_res
            timed_res.value
        end for s in 1:n_seed
    ]
    Zhat_LIS_dict[model_sym] = logsumexp.(getproperty.(resvec, :psis_logws)) .- log(10^6)
    ess_LIS_dict[model_sym] = getproperty.(resvec, :psis_logws) .|> compute_ess
end

Zhat_orig_dict = Dict{Symbol,Vector{Float64}}();
ess_orig_dict = Dict{Symbol,Vector{Float64}}();
for model_sym in model_syms
    resvec = [
        begin
            INFDIR = joinpath(@__DIR__, "output/seed$s");
            fname = joinpath(INFDIR, "orig_AMIS_$(model_sym).jld2");
            @load fname timed_res
            timed_res.value
        end for s in 1:n_seed
    ]
    Zhat_orig_dict[model_sym] = logsumexp.(getproperty.(resvec, :psis_logws)) .- log(10^6)
    ess_orig_dict[model_sym] = getproperty.(resvec, :psis_logws) .|> compute_ess
end

Zhat_rAMIS_dict = Dict{Symbol,Vector{Float64}}();
ess_rAMIS_dict = Dict{Symbol,Vector{Float64}}();
for model_sym in model_syms
    resvec = [
        begin
            INFDIR = joinpath(@__DIR__, "output/seed$s");
            fname = joinpath(INFDIR, "robust_AMIS_$(model_sym).jld2");
            @load fname timed_res
            timed_res.value
        end for s in 1:n_seed
    ]
    Zhat_rAMIS_dict[model_sym] = logsumexp.(getproperty.(resvec, :psis_logws)) .- log(10^6)
    ess_rAMIS_dict[model_sym] = getproperty.(resvec, :psis_logws) .|> compute_ess
end

Zhat_gold_dict = Dict(
    sym => logsumexp(Zhats)-log(n_seed) for (sym, Zhats) in Zhat_BS_dict
)

[std(Zhat_BS_dict[sym]) for sym in model_syms]

std_mat = hcat(
    [std(Zhat_LIS_dict[sym]) for sym in model_syms],
    [std(Zhat_orig_dict[sym]) for sym in model_syms],
    [std(Zhat_rAMIS_dict[sym]) for sym in model_syms],
    # [std(Zhat_alt_dict[sym]) for sym in model_syms],
    [std(Zhat_BS_dict[sym]) for sym in model_syms],
)'

bias_mat = hcat(
    [mean(Zhat_LIS_dict[sym]) - Zhat_gold_dict[sym] for sym in model_syms],
    [mean(Zhat_orig_dict[sym]) - Zhat_gold_dict[sym] for sym in model_syms],
    [mean(Zhat_rAMIS_dict[sym]) - Zhat_gold_dict[sym] for sym in model_syms],
    # [mean(Zhat_alt_dict[sym]) - Zhat_gold_dict[sym] for sym in model_syms],
    [mean(Zhat_BS_dict[sym]) - Zhat_gold_dict[sym] for sym in model_syms],
)'

sqrt.(abs2.(std_mat) .+ abs2.(bias_mat))

using Printf

row_labels = ["Laplace IS", "Standard AMIS", "Robust AMIS", "Bridge sampling"]

function fmt_sci(x)
    # Format as e-notation, then rewrite as LaTeX scientific notation
    s = @sprintf("%.2e", x)
    mantissa, exp_str = split(s, "e")
    exp_val = parse(Int, exp_str)           # strips leading zeros and "+"
    return "$(mantissa)\\times 10^{$(exp_val)}"
end

for i in 1:4
    bias_row = bias_mat[i,:]
    std_row  = std_mat[i,:]
    cells = join(["\\ftmath{$(fmt_sci(bval)) \\pm $(fmt_sci(sval))}"
                  for (bval, sval) in zip(bias_row, std_row)], " & ")
    println("$(row_labels[i]) & $(cells) \\\\")
end

summarystats(ess_rAMIS_dict[:richards])

[ess_LIS_dict[sym][1] for sym in model_syms]
[ess_orig_dict[sym][1] for sym in model_syms]
[ess_rAMIS_dict[sym][1] for sym in model_syms]

# Combine posteriors of carrying capactity K
INFDIR = joinpath(@__DIR__, "output/seed1");
all_K_samples = Dict(
    model_sym => begin 
        fname = joinpath(INFDIR, "robust_AMIS_$(model_sym).jld2");
        @load fname timed_res;
        psis_logws = timed_res.value.psis_logws
        sample(
            timed_res.value.all_samples[2,:], 
            weights(exp.(psis_logws .- maximum(psis_logws))), 
            10000, replace=true)
    end for model_sym in model_syms
);
logZvec_seed1 = [Zhat_rAMIS_dict[model_sym][1] for model_sym in model_syms]
BMA_ws = repeat(exp.(logZvec_seed1 .- maximum(logZvec_seed1)), inner=10000);
cat_K_samples = reduce(vcat, [all_K_samples[model_sym] for model_sym in model_syms]);
BMA_K_samples = sample(cat_K_samples, weights(BMA_ws), 10000, replace=true);


begin
    f = Figure(size=(1200, 800))

    use_log = false
    n_seed = 100;
    n_cat = 3;
    n_dodge = 4;
    cats = repeat(1:n_cat, inner=n_seed*n_dodge)
    dodges = repeat(repeat(1:n_dodge, inner=n_seed), n_cat)
    values = reduce(vcat, [
        vcat(
            Zhat_LIS_dict[sym], 
            Zhat_orig_dict[sym], 
            Zhat_rAMIS_dict[sym],
            Zhat_BS_dict[sym]
        ) .- Zhat_gold_dict[sym] for sym in model_syms
    ])
    base_colors = Makie.wong_colors()[[1, 3, 4, 2]]
    colors = base_colors[dodges]

    n = 2
    if use_log
        ytick_vals = [.-(0.1 .^ (0:n)); 0; 0.1 .^ (0:n)]
        ytick_labels = [
            v == 0 ? L"0" :
            begin
                exp = round(Int, log10(abs(v)))
                sign = v < 0 ? "-\\!" : ""
                L"%$(sign)10^{%$exp}"
            end
            for v in ytick_vals
        ]
    else
        ytick_vals = [-1, -0.6, -0.3, -0.1, -0.01, 0, 0.01, 0.1]
        ytick_labels = [
            begin
                val = abs(round(v) ≈ v ? Int(v) : v);
                sign = v < 0 ? "-\\!" : "";
                L"%$sign%$val"
            end for v in ytick_vals
        ]
    end
    ax = Axis(
        f[1:3,1], title="Log-evidence errors",
        titlesize=18,
        ylabel=L"\log\,\hat{Z} - \log\,Z_{\text{gold}}", ylabelsize=18,
        xticklabelsize=16,
        limits=((0.5, n_cat+0.5), use_log ? (-1.6, 1.6) : (nothing, nothing)),
        xticks=(1:3, ["Logistic\nmodel", "Gompertz\nmodel", "Richards'\nmodel"]),
        yscale=use_log ? Makie.Symlog10(1/10^n) : symsqrt,
        yticks=(ytick_vals, ytick_labels)
    )
    if use_log
        band!([0,n_cat+1], [-2, -2], [-0.01, -0.01], color=:grey, alpha=0.2)
        band!([0,n_cat+1], [0.01, 0.01], [2, 2], color=:grey, alpha=0.2)
    end
    lines!([0, n_cat+1], [0,0], color=:black, linestyle=:dash)
    # boxplot!(
    #     cats, values, dodge=dodges, color=colors, whiskerlinewidth=2,
    #     whiskercolor=base_colors[repeat(repeat(1:n_dodge, inner=8), n_cat)]
    # )   
    rainclouds!(
        cats, values, dodge=dodges, 
        color=[(c, 0.6) for c in colors], markersize=8,
        plot_boxplots=false, clouds=nothing, 
        gap=-0.1, dodge_gap=0.1,
        jitter_width=0.67
    ) 
    Legend(
        f[4, 1],
        [PolyElement(color = c, strokewidth = 0) for c in base_colors],
        ["Laplace IS", "Standard AMIS", "Robust AMIS", "Bridge sampling"],
        labelsize=16, orientation=:horizontal, nbanks=2, tellwidth=false
    )

    ax = Axis(
        f[1:3,2], title="Log-evidence errors",
        titlesize=18,
        xlabel="Effective sample size", xlabelsize=17,
        ylabel=L"\log\,\hat{Z} - \log\,Z_{\text{gold}}", ylabelsize=18,
        xticklabelsize=16,
        limits=((3e1,1.2e6), use_log ? (-1.6, 1.6) : (nothing, nothing)),
        xscale=log10,
        yscale=use_log ? Makie.Symlog10(1/10^n) : symsqrt,
        yticks=(ytick_vals, ytick_labels)
    )
    if use_log
        band!([0.1,10^7], [-2, -2], [-0.01, -0.01], color=:grey, alpha=0.2)
        band!([0.1,10^7], [0.01, 0.01], [2, 2], color=:grey, alpha=0.2)
    end

    lines!([1, 1e7], [0,0], color=:black, linestyle=:dash)

    # Second plot with models visually distinguished
    # markers = [:xcross, :cross, :circle]
    # for (ess_dict, Zhat_dict, color, msize) in zip(
    #     [ess_LIS_dict, ess_orig_dict, ess_rAMIS_dict],
    #     [Zhat_LIS_dict, Zhat_orig_dict, Zhat_rAMIS_dict],
    #     base_colors[1:3],
    #     (10, 11, 10)
    # )
    #     for (sym, marker) in zip(model_syms, markers)
    #         if marker === :circle
    #             msize *= 0.7
    #         elseif marker === :cross && ess_dict !== ess_LIS_dict
    #             msize *= 1.2
    #         end

    #         xcoords = ess_dict[sym]
    #         ycoords = Zhat_dict[sym] .- Zhat_gold_dict[sym]
    #         color_vec = [(color, (x < 8.3e5 || y > -1e-4) ? 0.5 : 0.02) for (x, y) in zip(xcoords, ycoords)]
    #         @info sym summarystats(last.(color_vec))
    #         scatter!(
    #             xcoords, ycoords,
    #             color=color_vec, marker=marker, markersize=msize
    #         )
    #     end
    # end
    
    # Legend(
    #     f[4, 2],
    #     [
    #         [MarkerElement(color=base_colors[c], marker=m, markersize=(c==2 ? 12 : 10)) for m in markers] 
    #         for c in 1:3
    #     ],
    #     [["Logistic", "Gompertz", "Richards'"] for _ in 1:3],
    #     ["Laplace IS", "Standard AMIS", "Robust AMIS"],
    #     labelsize=16, tellwidth=false, orientation=:horizontal, nbanks=3
    # )

    # Second plot without distinuigshing models visually
    markers = [:circle, :diamond, :xcross]
    for (ess_dict, Zhat_dict, color, marker, msize) in zip(
        [ess_LIS_dict, ess_orig_dict, ess_rAMIS_dict],
        [Zhat_LIS_dict, Zhat_orig_dict, Zhat_rAMIS_dict],
        base_colors[1:3], markers, (10, 12, 10)
    )
        for sym in model_syms
            xcoords = ess_dict[sym]
            ycoords = Zhat_dict[sym] .- Zhat_gold_dict[sym]
            # color_vec = [(color, (x < 8.3e5 || y > -1e-4) ? 0.5 : 0.02) for (x, y) in zip(xcoords, ycoords)]
            # @info sym summarystats(last.(color_vec))
            scatter!(
                xcoords, ycoords,
                color=(color, 0.6), marker=marker, markersize=msize
            )
        end
    end
    
    Legend(
        f[4, 2],
        [
            MarkerElement(color=base_colors[c], marker=m, markersize=(c==2 ? 12 : 10))
            for (c, m) in zip(1:3, markers)
        ],
        ["Laplace IS", "Standard\nAMIS", "Robust\nAMIS"],
        labelsize=16, tellwidth=false, orientation=:horizontal, 
    )


    n_cat = 5;
    ax = Axis(
        f[1, 3], title="Posteriors over models", titlesize=18,
        ylabel="Model posterior probability", ylabelsize=17,
        limits=(nothing, (0, 1)), yticks=0:0.2:1,
        xticks=(1:n_cat, ["BIC", "Laplace IS", "Standard AMIS", "Robust AMIS", "Bridge sampling"]),
        xticklabelrotation=π/6, 
        # xticks=(1:n_cat, ["BIC", "Laplace\nIS", "Standard\nAMIS", "Robust\nAMIS", "Bridge\nsampling"]),
        xticklabelsize=15
    )

    model_colors = Makie.wong_colors()[5:7]
    n_dodge = 3;
    cats = repeat(1:n_cat, inner=n_dodge)
    dodges = repeat(1:n_dodge, n_cat)
    values = reduce(vcat, [
        begin
            log_posts = [Zhat_dict[sym][1] for sym in model_syms];
            log_denom = logsumexp(log_posts);
            exp.(log_posts .- log_denom)
        end for Zhat_dict in [Zhat_BIC_dict, Zhat_LIS_dict, Zhat_orig_dict, Zhat_rAMIS_dict, Zhat_BS_dict]
    ])
    barplot!(cats, values, stack=dodges, color=model_colors[dodges])
    Legend(
        f[2, 3],
        [PolyElement(color = model_colors[c], strokewidth = 0) for c in 1:3],
        ["Logistic", "Gompertz", "Richards'"],
        labelsize=16, tellwidth=false, orientation=:horizontal
    )

    ax = Axis(
        f[3, 3], title="Posteriors of carrying capacity", titlesize=18,
        ylabel="Posterior density", ylabelsize=17,
        limits=((70, 95), (0, nothing)), yticks=0:0.1:1,
        xlabel=L"$K$ (% coral cover)", xlabelsize=17,
        xticklabelsize=15
    )
    for i in 1:3
        model_sym = model_syms[i]
        color = model_colors[i]
        kde_result = kde(all_K_samples[model_sym] .|> exp10)
        lines!(kde_result.x, kde_result.density, linewidth=2, color=color)
        # density!(
        #     all_K_samples[model_sym] .|> exp10, 
        #     # all_preds[model_sym],
        #     color=(:white, 0), strokewidth=2, strokecolor=color, strokearound=true, 
        # )
    end
    kde_result = kde(BMA_K_samples .|> exp10)
    lines!(kde_result.x, kde_result.density, linewidth=2, color=:grey20, linestyle=:dash)
    # density!(
    #     BMA_K_samples .|> exp10, 
    #     # BMA_preds,
    #     color=(:grey20, 0), strokewidth=2, strokecolor=:grey20, strokearound=true, linestyle=:dash
    # )
    Legend(
        f[4, 3],
        [
            [LineElement(color = model_colors[c]) for c in 1:3]; 
            LineElement(color = :grey20, linewidth=2, linestyle=:dash)
        ],
        ["Logistic", "Gompertz", "Richards'", "Model-averaged"],
        labelsize=16, tellwidth=false, orientation=:horizontal, nbanks=2,
    )

    # rowgap!(f.layout, 1, 0)
    # colsize!(f.layout, 3, Auto(0.7))

    for i in 1:4
        label = ["A", "B", "C", "D"][i]
        loc = [f[1, 1, TopLeft()], f[1, 2, TopLeft()], f[1, 3, TopLeft()], f[3, 3, TopLeft()]][i]
        rpad = [48, 48, 36, 32][i]
        Label(loc, label,
            fontsize = 24, font = :bold,
            padding = (0, rpad, 0, 0), # left, right, bottom, top
            halign = :right, valign = :center,
        )
    end

    display(f)
    save_dir = mkpath(joinpath(@__DIR__, "imgs/"));
    save("$(save_dir)/comparison_coral.png", f, px_per_unit=4);
end

# exit()

## Playground

# Zhat_LIS_dict[:richards]

# s = 1;
# INFDIR = joinpath(@__DIR__, "output/seed$s");
# mcmc_fname = joinpath(INFDIR, "chains_richards.jld2");
# @load mcmc_fname chn;

# d = 5;
# trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3]);
# samples = reshape(trace, d, :);
# X = samples[:,1:10:end];

# extremas = extrema.(eachrow(samples[:,1:10:end]))
# ax_limits = extremas

# # bad runs
# sym = :richards
# bad = sortperm(ess_rAMIS_dict[:richards])[1:10] # [40, 97, 74, 29, 28, 81, 43, 95, 54, 27]
# ess_rAMIS_dict[:richards][bad] # [1232, 1246, 1373, 1613, 1872, 2184, 3165, 3227, 3308, 4845]
# Zhat_rAMIS_dict[sym][bad] .- Zhat_gold_dict[sym] # [-0.08, -0.52, -0.48, -0.38, -0.35, -0.11, 0.05, -0.59, -0.39, -0.08]

# s = 40;
# INFDIR = joinpath(@__DIR__, "output/seed$s");
# fname = joinpath(INFDIR, "robust_AMIS_richards.jld2");
# @load fname timed_res;
# sc_color = :grey30

# f = plot_pairs(
#     # eachcol(exp10.(X)),
#     eachcol(X),
#     timed_res.value.gm_vec[1].means,
#     timed_res.value.gm_vec[1].chols .|> inv .|> Matrix,
#     title="Initial Gaussians (seed $s)", titlesize=17,
#     figsize=(120*d+60, 120*d), skip_upper=true,
#     scatter_kwargs=(color=sc_color, alpha=0.05, markersize=2),
#     # hexbin_kwargs=(colormap=Reverse(:grays), colorscale=log10),
#     ellipse_kwargs=(color=Makie.wong_colors()[3], alpha=0.2),
#     hist_kwargs=(color=:grey,),
#     axis_kwargs=(aspect=1,), 
#     hist_axis_kwargs=(aspect=1, yscale=log10,),
#     bins_vec=[range(a, b, 41) for (a, b) in extremas]
# ); 

# display(f)

# f = plot_pairs(
#     # eachcol(exp10.(X)),
#     eachcol(X),
#     timed_res.value.gm_vec[end].means,
#     timed_res.value.gm_vec[end].chols .|> inv .|> Matrix,
#     title="Final Gaussians (seed $s)", titlesize=17,
#     figsize=(120*d+60, 120*d), skip_upper=true,
#     scatter_kwargs=(color=sc_color, alpha=0.05, markersize=2),
#     # hexbin_kwargs=(colormap=Reverse(:grays), colorscale=log10),
#     ellipse_kwargs=(color=Makie.wong_colors()[3], alpha=0.3),
#     hist_kwargs=(color=:grey,),
#     axis_kwargs=(aspect=1,), 
#     hist_axis_kwargs=(aspect=1, yscale=log10,),
#     bins_vec=[range(a, b, 41) for (a, b) in extremas]
# ); 

# display(f)

# timed_res.value.psis_logws
# compute_ess(timed_res.value.psis_logws)
# timed_res.value.pareto_shape
# logsumexp(timed_res.value.psis_logws) - log(10^6)

# summarystats(Zhat_BS_dict[:richards])

# sqhdists = Float64[]
# mvnormals = [MvNormal(m, Matrix(inv(c))) for (m, c) in zip(timed_res.value.gm_vec[1].means, timed_res.value.gm_vec[1].chols)];
# K = timed_res.value.gm_vec[1].K
# @showprogress for k1 in 1:K
#     for k2 in (k1+1):K
#         push!(sqhdists, sqhdist_func(mvnormals[k1], mvnormals[k2]))
#     end
# end
# hist(sqhdists)
# viable_idxs = Int64[]
# for c in 1:K
#     is_viable = true
#     for v in viable_idxs
#         if sqhdist_func(mvnormals[c], mvnormals[v]) < 0.1
#             is_viable = false
#             break
#         end
#     end
#     is_viable && push!(viable_idxs, c)
# end
# length(viable_idxs)

# bad = [40, 97, 74, 29, 28, 81, 43, 95, 54, 27]
# for s in bad
#     INFDIR = joinpath(@__DIR__, "output/seed$s");
#     fname = joinpath(INFDIR, "robust_AMIS_richards.jld2");
#     @load fname timed_res;

#     essval = compute_ess(timed_res.value.psis_logws)
#     err = logsumexp(timed_res.value.psis_logws) - log(10^6) - (-41.51751182568784)

#     display((essval, err))
# end