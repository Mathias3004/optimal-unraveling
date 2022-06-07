using ITensors
using Distributed
using Statistics

#addprocs(128)


##################################

# includes
@everywhere begin
    
    include("TEData.jl")
    include("Functions.jl")
    include("TrajectorySampler.jl")

end



#function collect_data(ted::TEData, t_end::Float64, tau::Float64, dt::Float64)

# function to prepare TE data from input
function collect_data(N::Int64, Jx::Float64, Jz::Float64, gamma::Float64, optimal::Bool; 
        h::Float64=0.,
        verbose::Int64=1)::Dict{String,Any}

    dissipation = "Sz"

    maxdim = 700
    cutoff = 1E-8
    
    d_track = Dict{
    conserve_qns => true,
    track_states => false,
    track_maxdim => true,
    track_entropy => true,
    track_local_observables => ["Sz"],
    }


    # time evolution
    t_end = 2. # end time
    tau = 1. # for data collection
    dt = 0.1 # for discretization


    v_t = Array((0.: tau: t_end))
    
    var_keys = vcat(["maxdim"], track_local_observables)
    if track_entropy
        var_keys = vcat(["S"],var_keys)
    end
    
    # prepare TE
    params = XXZ_data(N, Jx, Jz, h, gamma, dissipation)
    ted = TE_data_XXZ(params, dt; maxdim=maxdim, cutoff=cutoff, conserve_qns=conserve_qns)

    # set initial state to half filling
    psi = productMPS(ted.s, n-> isodd(n) ? "Up" : "Dn")
    
    data_return = sample_trajectory(ted, psi, t_end, tau, optimal; d_tracks::Dict{String, Any}=Nothing)
    return data_return
end






