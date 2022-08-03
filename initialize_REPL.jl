
@time @everywhere push!(LOAD_PATH,"OptimalTrajectorySampler")
@everywhere using Revise
@time include("ParallelTrajectorySampler.jl")
include("run_XXZ_trajectories.jl")