using ITensors
using Statistics
using Revise
using Distributed

include("ParallelTrajectorySampler.jl")

struct XXZ_data
    N::Int64
    Jx::Float64
    Jz::Float64
    h::Float64
    gamma::Float64
    dissipation::String
end
    
    

function TE_data_XXZ(params::XXZ_data, dt::Float64; cutoff::Float64=1E-8, maxdim::Int64=300,conserve_qns::Bool=true)::TE_data

    s = siteinds("S=1/2", params.N; conserve_qns=conserve_qns)
    
    H_gates = get_H_XXZ(params.Jx, params.Jz, params.h, s, dt)
    c_gates, cdc_gates, Hcdc_gates = get_c(params.gamma, params.dissipation, s, dt)
    
    return TE_data(s, H_gates, c_gates, cdc_gates, Hcdc_gates, dt, cutoff, maxdim)
end

# prepare H gates
function get_H_XXZ(Jx::Float64, Jz::Float64, h::Float64, s::Vector{<:Index}, dt::Float64)::Array{ITensor,1}
    
    N = length(s)
    H_gates = ITensor[]

    # XXZ interaction
    for j in 1:(N-1)
        s1 = s[j]
        s2 = s[j + 1]
        hj = Jz*op("Sz", s1) * op("Sz", s2) +
             0.5*Jx *  (op("S+", s1) * op("S-", s2) + op("S-", s1) * op("S+", s2) )

        Gj = exp(-im * dt/2. * hj)
        push!(H_gates, Gj)
    end
    
    append!(H_gates, reverse(H_gates))
    
    # Sx field
    if abs(h) > 1E-4
        for s1 in s
            hj = -h * op( "Sx", s1)
            Gj = exp(-im * dt * hj)
            push!(H_gates, Gj)
            
        end
    end
    return H_gates
end

function get_c(gamma::Float64, dissipation::String, s::Vector{<:Index}, dt::Float64)
    N = length(s)
    
    c_gates = ITensor[]
    cdc_gates = ITensor[]
    Hcdc_gates = ITensor[]
    for j in 1:N
        c = sqrt(gamma) * op(dissipation, s[j])
        cd = replaceprime(dag(c), 0 => 2 )
        cdc = replaceprime(cd*c, 2 => 1 )
        Hcdc = exp( - 0.5*cdc*dt)
        
        push!(c_gates, c)
        push!(cdc_gates, cdc)
        push!(Hcdc_gates, Hcdc)
    end
    return c_gates, cdc_gates, Hcdc_gates
end


# function to prepare TE data from input
function collect_data_XXZ(
    N::Int64, 
    Jx::Float64,
    Jz::Float64, 
    gamma::Float64, 
    optimal::Bool, 
    t_end::Float64; 
    h::Float64=0.,
    tau::Float64=1.,
    dt::Float64=0.1,
    verbose::Int64=1,
    )

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
    
    collect_trajectories_synchronized(ted, psi, t_end, tau, optimal; d_tracks=d_tracks, pre="store_XXZ/dat_")
end







