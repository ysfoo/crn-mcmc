# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));

# E -> 0, 2A -> 0
data_idx = 12;

# This script takes one command-line argument, which is the dataset index.
# data_idx = parse(Int64, ARGS[1]);

# Fetch packages.
using Combinatorics, OrdinaryDiffEq, PEtab
using JLD2, ProgressMeter, Suppressor
using CairoMakie, LaTeXStrings
set_theme!(theme_latexfonts());
palette = Makie.wong_colors();

n_extra = length(rxs_extra);
n_models = length(models);

@load "params.jld2" param_sets
@load "data.jld2" all_data

data = all_data[data_idx];
petab_models = @showprogress [create_petab_model(model, data, u0) for model in models];
@load "MLE_fits/data$(data_idx).jld2" model_fits MLE_hessians;
petab_probs = @showprogress [PEtabODEProblem(m) for m in petab_models];

# Inspect model fit for data-generating model
model_idx = data_idx;
model_fit = model_fits[model_idx];
model_fit.fmin
petab_model = petab_models[model_idx];
petab_prob = petab_probs[model_idx];
fit_sol = PEtab.get_odesol(model_fit.xmin, petab_prob);

# Integer combiantions
combs = [Int64[], combinations(collect(1:n_extra))...];
rx_boolmat = [Int(rx ∈ comb) for comb in combs, rx in 1:n_extra] # 2^R by R

# Alternative fit
other_idx = 21; # replace 
other_fit = model_fits[other_idx];
other_fit.fmin
other_model = petab_models[other_idx];
other_sol = PEtab.get_odesol(other_fit.xmin, PEtabODEProblem(other_model));

# Compare model fits to ground truth
t_save = range(0, t_final, 101);
fit_solmat = reduce(hcat, fit_sol(t_save).u);
other_solmat = reduce(hcat, other_sol(t_save).u);

oprob = ODEProblem(models[data_idx], u0, data.t[end], param_sets[data_idx]);
gt_sol = solve(oprob);
gt_solmat = reduce(hcat, gt_sol(t_save).u);

# Latex labels
rx_labels = [
        begin 
        s = string(rx)
        s = s[findfirst(' ', s)+1:end]
        s = Base.replace(s, "-->" => "\\rightarrow")
        s = Base.replace(s, "X" => "X_")
        s = Base.replace(s, "*" => "")
        s
    end for rx in rxs_extra
]


# Plot -log p(y | θ, model) @ MAP
nllhs = @showprogress [petab_prob.nllh(model_fit.xmin; prior=false) for (petab_prob, model_fit) in zip(petab_probs, model_fits)];
begin
    f = Figure()
    ax = Axis(
        f[1,1], xlabel="Model index", ylabel=L"p(\mathbf{y} \mid \theta, \text{model})",
        xminorticks=8:16:64, xminorgridvisible=true, xminorgridcolor=RGBAf(0, 0, 0, 0.12),
        limits=((0.5, n_models+0.5), nothing), xticks=16:16:64,
    )
    scatter!(sort(nllhs))
    
    ax = Axis(
        f[2,1], ylabel="Reaction inclusion",
        limits=((0.5, n_models+0.5), (0.5, n_extra+0.5)), xticks=16:16:64, 
        xgridvisible=false, xaxisposition=:top, yreversed=true, ygridvisible=false, 
        yticks=(1:6, latexstring.(rx_labels)),
    )
    for (i, m) in enumerate(sortperm(nllhs))
        for j in combs[m]
            band!(
                [i-0.5, i+0.5], fill(j-0.5,2), fill(j+0.5,2),
                color = Makie.wong_colors()[(j ∈ combs[data_idx]) ? 5 : 6]
            )
        end
    end
    rx_str = join(rx_labels[combs[data_idx]], ",\\; ")
    Label(f[0,1], latexstring("\\textbf{Dataset $data_idx:} \$$rx_str\$"), tellwidth=false, fontsize=16)
    rowsize!(f.layout, 1, Relative(0.6))
    rowgap!(f.layout, 2, 5)
    f
end

# Plot -log p(y, θ | model) @ MAP
fmins = getproperty.(model_fits, :fmin);
begin
    f = Figure()
    ax = Axis(
        f[1,1], xlabel="Model index", ylabel=L"p(\mathbf{y}, \theta \mid \text{model})",
        xminorticks=8:16:64, xminorgridvisible=true, xminorgridcolor=RGBAf(0, 0, 0, 0.12),
        limits=((0.5, n_models+0.5), nothing), xticks=16:16:64,
    )
    scatter!(sort(fmins))
    
    ax = Axis(
        f[2,1], ylabel="Reaction inclusion",
        limits=((0.5, n_models+0.5), (0.5, n_extra+0.5)), xticks=16:16:64, 
        xgridvisible=false, xaxisposition=:top, yreversed=true, ygridvisible=false, 
        yticks=(1:6, latexstring.(rx_labels)),
    )
    for (i, m) in enumerate(sortperm(fmins))
        for j in combs[m]
            band!(
                [i-0.5, i+0.5], fill(j-0.5,2), fill(j+0.5,2),
                color = Makie.wong_colors()[(j ∈ combs[data_idx]) ? 5 : 6]
            )
        end
    end
    rx_str = join(rx_labels[combs[data_idx]], ",\\; ")
    Label(f[0,1], latexstring("\\textbf{Dataset $data_idx:} \$$rx_str\$"), tellwidth=false, fontsize=16)
    rowsize!(f.layout, 1, Relative(0.6))
    rowgap!(f.layout, 2, 5)
    f
end


# All optimised BIC
nparams = length.(combs) .+ 4
BICs = 2nllhs .+ nparams .* log(length(u0)*length(data.t))
begin
    f = Figure()
    ax = Axis(
        f[1,1], xlabel="Model index", ylabel="BIC",
        xminorticks=8:16:64, xminorgridvisible=true, xminorgridcolor=RGBAf(0, 0, 0, 0.12),
        limits=((0.5, n_models+0.5), nothing), xticks=16:16:64,
    )
    scatter!(sort(BICs))
    
    ax = Axis(
        f[2,1], ylabel="Reaction inclusion",
        limits=((0.5, n_models+0.5), (0.5, n_extra+0.5)), xticks=16:16:64, 
        xgridvisible=false, xaxisposition=:top, yreversed=true, ygridvisible=false, 
        yticks=(1:6, latexstring.(rx_labels)),
    )
    for (i, m) in enumerate(sortperm(BICs))
        for j in combs[m]
            band!(
                [i-0.5, i+0.5], fill(j-0.5,2), fill(j+0.5,2),
                color = Makie.wong_colors()[(j ∈ combs[data_idx]) ? 5 : 6]
            )
        end
    end
    rx_str = join(rx_labels[combs[data_idx]], ",\\; ")
    Label(f[0,1], latexstring("\\textbf{Dataset $data_idx:} \$$rx_str\$"), tellwidth=false, fontsize=16)
    rowsize!(f.layout, 1, Relative(0.6))
    rowgap!(f.layout, 2, 5)
    f
end

# Laplace approximation
using LinearAlgebra

logdets = -logdet.(MLE_hessians);
LAs = -fmins .+ 0.5logdets + 0.5nparams*log(2π)
begin
    f = Figure()
    ax = Axis(
        f[1,1], xlabel="Model index", ylabel=L"Laplace approx of $-p(\mathbf{y} \mid \text{model})$",
        xminorticks=8:16:64, xminorgridvisible=true, xminorgridcolor=RGBAf(0, 0, 0, 0.12),
        limits=((0.5, n_models+0.5), nothing), xticks=16:16:64,
    )
    scatter!(sort(-LAs))
    
    ax = Axis(
        f[2,1], ylabel="Reaction inclusion",
        limits=((0.5, n_models+0.5), (0.5, n_extra+0.5)), xticks=16:16:64, 
        xgridvisible=false, xaxisposition=:top, yreversed=true, ygridvisible=false, 
        yticks=(1:6, latexstring.(rx_labels)),
    )
    for (i, m) in enumerate(sortperm(-LAs))
        for j in combs[m]
            band!(
                [i-0.5, i+0.5], fill(j-0.5,2), fill(j+0.5,2),
                color = Makie.wong_colors()[(j ∈ combs[data_idx]) ? 5 : 6]
            )
        end
    end
    rx_str = join(rx_labels[combs[data_idx]], ",\\; ")
    Label(f[0,1], latexstring("\\textbf{Dataset $data_idx:} \$$rx_str\$"), tellwidth=false, fontsize=16)
    rowsize!(f.layout, 1, Relative(0.6))
    rowgap!(f.layout, 2, 5)
    f
end


# Strucutral uncertainty via BIC
using LogExpFunctions, StatsBase
logposts_BIC = -0.5BICs;
denom_BIC = logsumexp(logposts_BIC)
ws_BIC = ProbabilityWeights(exp.(logposts_BIC .- denom_BIC));
rx_probs_BIC = rx_boolmat' * ws_BIC
cmat_BIC = cor(rx_boolmat, ws_BIC)

begin
    f = Figure(size=(700, 400), figure_padding=(16, 16, 32, 8))
    a1 = Axis(
        f[1,1], xscale=log10, xautolimitmargin=(0.,0.05), xreversed=true, yaxisposition=:right,
        yautolimitmargin=(0.,0.), ylabel="Reaction index", #xlabel="Posterior prob.",
        yticks=1:n_extra, ylabelsize=16,
        xticklabelsize=16, yticklabelsize=16,
        limits=(nothing, (0.5, n_extra+0.5)), 
        title="Posterior probabilities", titlesize=16, titlefont=:regular
    )
    a2 = Axis(
        f[1,2], aspect=DataAspect(), xlabel="Reaction index",
        yticks=1:n_extra, yticklabelsvisible=false,
        xticks=1:n_extra, xlabelsize=16, xticklabelsize=16,
        title="Posterior correlations", titlesize=16, titlefont=:regular
    )
    linkyaxes!(a1, a2)
    barplot!(
        a1, rx_probs_BIC, direction=:x,
        color=[palette[r∈combs[data_idx] ? 2 : 1] for r in 1:n_extra]
    )
    Legend(
        f[1,1],
        [PolyElement(color=palette[2]), PolyElement(color=palette[1])],
        ["Ground truth", "Spurious"],
        gridshalign=:left, labelsize=14,
        orientation=:horizontal,
        tellheight = false, halign=:center,
        tellwidth = false, valign=:bottom,
        margin = (10, 10, -60, 10), 
        patchsize=(15.,15.),
        # padding = (6, 6, 6, 6),
    )
    hm = heatmap!(a2, cmat_BIC, colormap=:RdBu, colorrange=(-1,1))
    # cscale = Makie.pseudolog10
    # cticks = [-1, -0.5, -0.2, 0, 0.2, 0.5, 1]
    cscale = identity
    cticks = [-1, -0.5, 0, 0.5, 1]
    hm = heatmap!(a2, cmat_BIC, colormap=:RdBu, colorrange=(-1,1), colorscale=cscale)
    Colorbar(
        f[1,3], hm, ticklabelsize=16, scale=cscale,
        ticks = cticks
    )
    Label(f[0, :], "CRN posterior using BIC approximation", font=:bold, fontsize=18)
    rowgap!(f.layout, 1, 5)
    colgap!(f.layout, 1, 0)
    colgap!(f.layout, 2, 5)
    f
end


# Strucutral uncertainty via Laplace approx
logposts_LA = LAs;
denom_LA = logsumexp(logposts_LA)
ws_LA = ProbabilityWeights(exp.(logposts_LA .- denom_LA))
rx_probs_LA = rx_boolmat' * ws_LA
cmat_LA = cor(rx_boolmat, ws_LA)

begin
    f = Figure(size=(700, 400), figure_padding=(16, 16, 32, 8))
    a1 = Axis(
        f[1,1], xscale=log10, xautolimitmargin=(0.,0.05), xreversed=true, yaxisposition=:right,
        yautolimitmargin=(0.,0.), ylabel="Reaction index", #xlabel="Posterior prob.",
        yticks=1:n_extra, ylabelsize=16,
        xticklabelsize=16, yticklabelsize=16,
        limits=(nothing, (0.5, n_extra+0.5)), 
        title="Posterior probabilities", titlesize=16, titlefont=:regular
    )
    a2 = Axis(
        f[1,2], aspect=DataAspect(), xlabel="Reaction index",
        yticks=1:n_extra, yticklabelsvisible=false,
        xticks=1:n_extra, xlabelsize=16, xticklabelsize=16,
        title="Posterior correlations", titlesize=16, titlefont=:regular
    )
    linkyaxes!(a1, a2)
    barplot!(
        a1, rx_probs_LA, direction=:x,
        color=[palette[r∈combs[data_idx] ? 2 : 1] for r in 1:n_extra]
    )
    Legend(
        f[1,1],
        [PolyElement(color=palette[2]), PolyElement(color=palette[1])],
        ["Ground truth", "Spurious"],
        gridshalign=:left, labelsize=14,
        orientation=:horizontal,
        tellheight = false, halign=:center,
        tellwidth = false, valign=:bottom,
        margin = (10, 10, -60, 10), 
        patchsize=(15.,15.),
        # padding = (6, 6, 6, 6),
    )
    # cscale = Makie.pseudolog10
    # cticks = [-1, -0.5, -0.2, 0, 0.2, 0.5, 1]
    cscale = identity
    cticks = [-1, -0.5, 0, 0.5, 1]
    hm = heatmap!(a2, cmat_LA, colormap=:RdBu, colorrange=(-1,1), colorscale=cscale)
    Colorbar(
        f[1,3], hm, ticklabelsize=16, scale=cscale,
        ticks = cticks
    )
    Label(f[0, :], "CRN posterior using Laplace approximation", font=:bold, fontsize=18)
    rowgap!(f.layout, 1, 5)
    colgap!(f.layout, 1, 0)
    colgap!(f.layout, 2, 5)
    f
end





# Temporary
petab_probs[end].nllh.(getproperty.(model_fits[end].runs, :xmin))
getproperty.(model_fits[end].runs, :xmin)
sortperm(fmins)

petab_probs[12].nllh.(getproperty.(model_fits[12].runs, :xmin))

petab_probs[31].nllh.(getproperty.(model_fits[31].runs, :xmin))
reduce(hcat, collect.(getproperty.(model_fits[31].runs, :xmin)))'
petab_models[31].parametermap

begin
    f = Figure(size=(800, 450))
    for d in eachindex(u0)
        ax = Axis(f[1, d])
        lines!(t_save, gt_solmat[d,:], color=Makie.wong_colors()[d])
        lines!(t_save, fit_solmat[d,:], color=Makie.wong_colors()[d], linestyle=:dash)
        lines!(t_save, other_solmat[d,:], color=Makie.wong_colors()[d], linestyle=:dot)
    end
    Legend(
        f[2, :],
        [LineElement(;linestyle) for linestyle in (:solid, :dash, :dot)],
        ["Ground truth", "Fit of data-generating model", "Fit of model with alternative death term"],
        tellwidth=false, orientation = :horizontal
    )
    # rowsize!(f.layout, 4, Relative(0.1))
    f
end

