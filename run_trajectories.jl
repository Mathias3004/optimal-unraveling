using ITensors
using Statistics

#addprocs(128)


##################################

include("TEData.jl")
include("Functions.jl")
include("TrajectorySampler.jl")




#function collect_data(ted::TEData, t_end::Float64, tau::Float64, dt::Float64)

# function to prepare TE data from input
function test_collect_XXZ(N::Int64, Jx::Float64, Jz::Float64, gamma::Float64, optimal::Bool, t_end::Float64; 
        h::Float64=0.,
        tau::Float64=1.,
        dt::Float64=0.1,
        verbose::Int64=1,
        )::Dict{String,Any}

    dissipation = "Sz"

    maxdim = 700
    cutoff = 1E-8
    
    d_tracks = Dict(
        "track_states" => false,
        "track_maxdim" => true,
        "track_entropy" => true,
        "track_local_observables" => ["Sz"]
    )


    v_t = Array((0.: tau: t_end))
    
    # prepare TE
    params = XXZ_data(N, Jx, Jz, h, gamma, dissipation)
    ted = TE_data_XXZ(params, dt; maxdim=maxdim, cutoff=cutoff, conserve_qns=true)

    # set initial state to half filling
    psi = productMPS(ted.s, n-> isodd(n) ? "Up" : "Dn")
    
    data_return = sample_trajectory(ted, psi, t_end, tau, optimal; d_tracks=d_tracks)
    return data_return
end






