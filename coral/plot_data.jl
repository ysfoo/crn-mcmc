include(joinpath(@__DIR__, "setup.jl"));

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, PEtab, Random
using JLD2, ProgressMeter, Suppressor

include(joinpath(@__DIR__, "../plot_helpers.jl"));

begin
    f = Figure()
    ax = Axis(
        f[1,1],
        xlabel="Time (days)", ylabel="Hard coral cover (%)",
        xlabelsize=18, ylabelsize=18,
        xticklabelsize=16, yticklabelsize=16
    )
    scatter!(times, obs)
    display(f)
    save_dir = mkpath(joinpath(@__DIR__, "imgs/"))
    mkpath(save_dir)
    save("$(save_dir)/coral_data.png", f, px_per_unit=4);
end

