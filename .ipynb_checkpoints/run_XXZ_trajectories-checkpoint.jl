using ITensors
using Statistics
using Revise
using Distributed
using Printf

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
    optimal::String, 
    t_end::Float64; 
    dissipation::String = "Sz",
    h::Float64=0.,
    tau::Float64=1.,
    dt::Float64=0.1,
    d_tracks::Dict{String,Any}=Nothing,
    d_controls::Dict{String,Any}=Nothing,
    verbose::Int64=1,
    pre="store_XXZ/dat_",
    append=false
    )

    # MPS controls
    if d_controls == Nothing
        maxdim = 700
        cutoff = 1E-8
        conserve_qns = true
    else
        maxdim = d_controls["maxdim"]
        cutoff = d_controls["cutoff"]
        conserve_qns = d_controls["conserve_qns"]
    end
    
    # time array for data collection
    v_t = Array((0.: tau: t_end))
    
    # prepare H and c's for time evolution XXZ model
    params = XXZ_data(N, Jx, Jz, h, gamma, dissipation)
    ted = TE_data_XXZ(params, dt; maxdim=maxdim, cutoff=cutoff, conserve_qns=conserve_qns)

    # set initial state to half filling
    psi = productMPS(ted.s, n-> isodd(n) ? "Up" : "Dn")
    
    # sample trajectories on parallel threads, sync after each time step
    collect_trajectories_synchronized(ted, psi, t_end, tau, optimal; d_tracks=d_tracks, pre=pre, append=append)
end

######## MAIN ################

function main(args_in)

    # define parameters
    
    # command line input
    N = parse(Int64, args_in[1])
    gamma = parse(Float64, args_in[2])
    Jz = parse(Float64, args_in[3]) # Jz coupling (dipole-dipole)
    h = parse(Float64, args_in[4]) # magnetic field
    optimization = args_in[5]
    dir = args_in[6]
    
    # fixed params
    Jx = -1. # Jx coupling (flip flop) set to unity (ferromagnetic)
    dissipation = "projUp" # the type of dissipation
    
    t_end = 500. # the total time to evolve
    tau = 1. # the time step to collect data
    dt = 0.02 # differential time step for integration
    
    pre_store = dir * "/dat__opt_" * optimization # the folder + prefix where to store data
    
    # variables to track and save
    d_tracks = Dict{String,Any}(
        "track_states" => false, # whether to save all the sampled states (better not for memory!)
        "track_maxdim" => true, # track maxdim in MPS
        "track_entropy" => true, # save entropy profile at each time step
        "track_local_observables" => ["Sz"] # profile of local observables to save at each time step
    )
    
    # controls for MPS
    d_controls = Dict{String,Any}(
        "maxdim" => 1000, # max bond dim during simulation
        "cutoff" => 1E-8, # sv cutoff
        "conserve_qns" => true # U(1) symmetry from particle conservation (e.g. with Sz dissipation) or not, more efficient simulation
    )
    
    # run parallel trajectories on number of available workers
    println("Collecting data points:")
    optimal = optimization
    pre = pre_store * @sprintf("_N_%.0f_Jx_%.2f_Jz_%.2f_gamma_%.2f", N, Jx, Jz, gamma) # prefix to store the data in txt file, will be "_it_
    collect_data_XXZ(
        N, 
        Jx,
        Jz, 
        gamma, 
        optimal, 
        t_end; 
        dissipation=dissipation,
        h=h,
        tau=tau,
        dt=dt,
        pre=pre,
        d_tracks=d_tracks,
        d_controls=d_controls,
        verbose=1,
        append=false
    )
        
 
end

# set main() as standard function to run if not specified otherwise
if abspath(PROGRAM_FILE) == @__FILE__

    @time @everywhere push!(LOAD_PATH,"OptimalTrajectorySampler")
    @time include("ParallelTrajectorySampler.jl")

    @time main(ARGS)
end
    







