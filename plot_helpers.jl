### Plotting helpers

using CairoMakie, LaTeXStrings, LinearAlgebra
set_theme!(theme_latexfonts());
update_theme!(
	Axis=(;
		xgridvisible=false, ygridvisible=false,
	),
)

function add_ellipse!(ax, meanvec, covmat, i1, i2; kwargs...)
    thres = quantile(Chisq(length(meanvec)), 0.95)
    eigenvals, eigenvecs = eigen(covmat[[i1,i2],[i1,i2]])
    semi_axes = sqrt.(eigenvals * thres)
    θ = atan(eigenvecs[2, 1], eigenvecs[1, 1])
    tgrid = range(0, 2π, length=200)
    ellipse_unit = hcat(cos.(tgrid), sin.(tgrid))
    ellipse_scaled = ellipse_unit * diagm(semi_axes)
    rotation_matrix = [cos(θ) -sin(θ);
                        sin(θ)  cos(θ)]
    ellipse_rotated = ellipse_scaled * rotation_matrix'
    ellipse = ellipse_rotated .+ [meanvec[i1] meanvec[i2]]
    lines!(ax, ellipse; kwargs...)
end

function plot_pairs(
    states, μs=[], Σs=[]; 
    title=nothing, figsize=nothing, axis_kwargs=(;),
    scatter_kwargs=(;markersize=5, alpha=0.5), 
    ellipse_kwargs=(;),
    hist_kwargs=(;),
)
    n_dim = length(states[1])
    if isnothing(figsize)
        figsize = (120*n_dim, 120*n_dim)
    end
    f = Figure(size=figsize)
    for i1 in 1:n_dim
        for i2 in 1:n_dim
            if i1 == i2
                ax = Axis(f[i2,i1]; axis_kwargs..., yscale=log10)
                hist!(getindex.(states, i1); bins=50, normalization=:pdf, hist_kwargs...)
            else
                ax = Axis(f[i2,i1]; axis_kwargs...)
                scatter!(
                    getindex.(states, i1), getindex.(states, i2); scatter_kwargs...
                )
                autolimits!(ax)
                ax_limits = ax.finallimits[]
                for (μ, Σ) in zip(μs, Σs)
                    add_ellipse!(ax, μ, Σ, i1, i2; ellipse_kwargs...)
                end
                limits!(ax, ax_limits)
            end

            if i1 ∈ [1, n_dim]
                ax.yaxisposition = i1 == 1 ? :left : :right
                ax.yticklabelsize = 12
            else
                ax.yticksvisible = false
                ax.yticklabelsvisible = false
            end
            
            if i2 ∈ [1, n_dim]
                ax.xaxisposition = i2 == 1 ? :top : :bottom
                ax.xticklabelrotation = π/4
                ax.xticklabelsize = 12
            else
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
            end
        end
    end
    if !isnothing(title) && length(title) > 0
        Label(f[0,:], title, fontsize=16)
    end
    return f
end