include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../plot_helpers.jl"));

using Turing, MCMCChains
using PDMats, LogExpFunctions

MAP_fname = joinpath(@__DIR__, "output/MAPs.jld2");
@load MAP_fname model_fits;

# Posterior plots

seed = 1;
INFDIR = joinpath(@__DIR__, "output/seed$(seed)");

syms = [:logistic, :richards]
sc_color = :grey30
multif = Figure(size = (1200, 675));
for fidx in 1:2
    f = multif[1, fidx] = GridLayout()
    model_sym = syms[fidx]
    name = name_dict[model_sym]
    d = nparam_dict[model_sym]
    ps = psym_dict[model_sym]
    prior_dists = priors_dict[model_sym];
    MAP = model_fits[model_sym].value.MAP;
    hess = model_fits[model_sym].value.hess;
    Σ = inv(PDMat(hermitianpart!(hess)));
    
    mcmc_fname = joinpath(INFDIR, "chains_$(model_sym).jld2");
    @load mcmc_fname chn;
    trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3]);
    samples = reshape(trace, d, :);
    X = samples[:,1:1:end];

    extremas = extrema.(eachrow(samples[:,1:15:end]))
    ax_limits = extremas
    
    f = plot_pairs(
        # eachcol(exp10.(X)),
        eachcol(X),
        [MAP], [Σ],
        title="Posterior samples under $name model", titlesize=17,
        layout=f, skip_upper=true,
        scatter_kwargs=(color=sc_color, alpha=0.008, markersize=2),
        # hexbin_kwargs=(colormap=Reverse(:grays), colorscale=log10),
        ellipse_kwargs=(color=Makie.wong_colors()[3],),
        hist_kwargs=(color=:grey,),
        axis_kwargs=(aspect=1,),
        hist_axis_kwargs=(aspect=1, yscale=log10,),
        bins_vec=[range(a, b, 41) for (a, b) in extremas]
    );    

    idx = 0
    for (i1, p1) in enumerate(ps) # which row
        for (i2, p2) in enumerate(ps) # which column
            if i1 < i2
                continue
            end
            idx += 1
            ax = f.content[idx].content     
            if i1 == i2
                autolimits!(ax)
                fl = ax.finallimits[]
                i = i1
                dist = prior_dists[i]
                xs = dist.μ-5dist.σ:0.05:dist.μ+5dist.σ
                lines!(ax, xs, pdf.(Ref(prior_dists[i]), xs), color=Makie.wong_colors()[1])
                ylow, yheight = fl.origin[2], fl.widths[2]
                new_ylow = ylow*((ylow+yheight)/ylow)^(1/21)
                limits!(ax, ax_limits[i2], (new_ylow, fl.origin[2]+fl.widths[2]))
            else
                # dist = prior_dists[i2]
                # xs = dist.μ-3dist.σ:0.05:dist.μ+3dist.σ
                # dist = prior_dists[i1]
                # ys = dist.μ-3dist.σ:0.05:dist.μ+3dist.σ
                limits!(ax, ax_limits[i2], ax_limits[i1])
            end
            if i1 == i2
                ax.yaxisposition = :right
                ax.yticklabelsize = 11
                ax.yticklabelpad = 0.5
                ax.yticksvisible = true
                ax.yticklabelsvisible = true
            elseif i2 ∈ [1, d]
                ax.yaxisposition = i2 == 1 ? :left : :right
                ax.ylabel = L"\log_{10} %$(sym2label[Symbol(p1)])"
                ax.ylabelsize = 16
                ax.yticklabelsize = 12
                ax.yticksvisible = true
                ax.yticklabelsvisible = true
            else
                ax.yticksvisible = false
                ax.yticklabelsvisible = false
            end
            
            if i1 ∈ [d]
                ax.xaxisposition = i1 == 1 ? :top : :bottom
                ax.xlabel = L"\log_{10} %$(sym2label[Symbol(p2)])"
                # ax.xlabel = L"%$(sym2label[Symbol(p2)])"
                ax.xlabelsize = 16
                ax.xticklabelrotation = π/4
                ax.xticklabelsize = 12
            else
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
            end
        end
    end

    colgap!(f, [-8, -10][fidx])
end

Legend(
    multif[2,:], 
    [
        LineElement(color = Makie.wong_colors()[1], ),
        PolyElement(color = :grey, strokewidth = 0, points = Point2f[(0, 0.25), (1, 0.25), (1, 0.75), (0, 0.75)]),
        MarkerElement(color = sc_color, marker=:circle, markersize=8), 
        LineElement(color = Makie.wong_colors()[3], ),
    ],
    ["Prior density", "Posterior density", "Posterior samples", "Laplace approximation"],
    labelsize=18, orientation=:horizontal   
);

for fidx in 1:2
    Label(multif[1, fidx][0, 1, Left()], ["A", "B"][fidx],
        fontsize = 24,
        font = :bold,
        padding = (0, 30, 0, 0), # left, right, bottom, top
        halign = :right,
        valign = :center,
    )
end

display(multif)

save_dir = mkpath(joinpath(@__DIR__, "imgs/post_logparams/"));
save("$(save_dir)/logistic_richards.png", multif, px_per_unit=4);