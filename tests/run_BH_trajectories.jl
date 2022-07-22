using ITensors
using Statistics
using Revise
using Distributed
using Printf

struct BH_data
    N::Int64
    J::Float64
    V::Float64
    V2::Float64
    gamma::Float64
    dissipation::String
end
    
    

function TE_data_BH(params::BH_data, dt::Float64; cutoff::Float64=1E-8, maxdim::Int64=300,conserve_qns::Bool=true, dim::Int64=2)::TE_data

    s = siteinds("Qudit", params.N; conserve_qns=conserve_qns, dim=dim)
    
    H_gates = get_H_BH(params.J, params.V, params.V2, s, dt)
    c_gates, cdc_gates, Hcdc_gates = get_c(params.gamma, params.dissipation, s, dt)
    
    return TE_data(s, H_gates, c_gates, cdc_gates, Hcdc_gates, dt, cutoff, maxdim)
end

# prepare H gates
function get_H_BH(J::Float64, V::Float64, V2::Float64, s::Vector{<:Index}, dt::Float64)::Array{ITensor,1}
    
    N = length(s)
    H_gates = ITensor[]

    # tunneling and nearest neighbor interaction
    for j in 1:(N-1)
        s1 = s[j]
        s2 = s[j + 1]
        hj = 0.5*J *  (op("adag", s1) * op("a", s2) + op("a", s1) * op("adag", s2) )
            + V*op("n", s1) * op("n", s2) 
        Gj = exp(-im * dt/2. * hj)
        push!(H_gates, Gj)
    end
    
    # next nearest neighbor interaction
    if abs(V2) > 1E-4
        for j in 1:(N-2)
            hj = V2 * op("n", s[j]) * op("n", s[j+2])
            Gj = exp(-im * dt/2. * hj)
            push!(H_gates, Gj)
        end
    end
    
    # add reverse array to have time step dt
    append!(H_gates, reverse(H_gates))
    
    return H_gates
end

# array of dissipation operators
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
function collect_data_BH(
    N::Int64, 
    J::Float64,
    V::Float64, 
    gamma::Float64, 
    optimal::Bool, 
    t_end::Float64; 
    dissipation::String = "Sz",
    V2::Float64=0.,
    tau::Float64=1.,
    dt::Float64=0.1,
    d_tracks::Dict{String,Any}=Nothing,
    d_controls::Dict{String,Any}=Nothing,
    verbose::Int64=1,
    pre="store_BH/dat_",
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
    params = BH_data(N, J, V, V2, gamma, dissipation)
    ted = TE_data_BH(params, dt; maxdim=maxdim, cutoff=cutoff, conserve_qns=conserve_qns)

    # set initial state to half filling
    psi = productMPS(ted.s, n-> isodd(n) ? "1" : "0")
    
    # sample trajectories on parallel threads, sync after each time step
    collect_trajectories_synchronized(ted, psi, t_end, tau, optimal; d_tracks=d_tracks, pre=pre, append=append)
end

######## MAIN ################

function main(args_in)

    # define parameters
    
    # command line input
    N = parse(Int64, args_in[1])
    gamma = parse(Float64, args_in[2])
    V = parse(Float64, args_in[3]) # Jz coupling (dipole-dipole)
    V2 = parse(Float64, args_in[4]) # magnetic field
    dir = args_in[5]
    
    # fixed params
    J = 1. # Jx coupling (flip flop)
    dissipation = "n" # the type of dissipation
    
    t_end = 20. # the total time to evolve
    tau = 1. # the time step to collect data
    dt = 0.05 # differential time step for integration
    
    pre_store = dir * "/dat_" # the folder + prefix where to store data
    
    n_runs = 2 # number of times you want to generate the same trajectories. Each run nworkers samples are collected and saved
    
    # variables to track and save
    d_tracks = Dict{String,Any}(
        "track_states" => false, # whether to save all the sampled states (better not for memory!)
        "track_maxdim" => true, # track maxdim in MPS
        "track_entropy" => true, # save entropy profile at each time step
        "track_local_observables" => ["n"] # profile of local observables to to save at each time step
    )
    
    # controls for MPS
    d_controls = Dict{String,Any}(
        "maxdim" => 700, # max bond dim during simulation
        "cutoff" => 1E-8, # sv cutoff
        "conserve_qns" => true # U(1) symmetry from particle conservation (e.g. with Sz dissipation) or not, more efficient simulation
    )
    
    # run parallel trajectories on number of available workers and loop over number of runs to collect
    for i_n = 1:n_runs
        println("\nCollecting run $(i_n)/$(n_runs):\n")
        append = i_n != 1 # set first run to write (start new files), append to files in next runs to collect more samples
        
        # using 2x2 optimized jump clicks
        println("Optimized dissipation:")
        optimal = true
        pre = pre_store * @sprintf("_opt_N_%.0f_J_%.2f_V_%.2f_gamma_%.2f", N, J, V, gamma) # prefix to store the data in txt file, will be "_it_
        collect_data_BH(
            N, 
            J,
            V, 
            gamma, 
            optimal, 
            t_end; 
            dissipation=dissipation,
            V2=V2,
            tau=tau,
            dt=dt,
            pre=pre,
            d_tracks=d_tracks,
            d_controls=d_controls,
            verbose=1,
            append=append
        )
        
        # using simple local dissipation
        println("Local dissipation:")
        optimal = false
        pre = pre_store * @sprintf("_loc_N_%.0f_J_%.2f_V_%.2f_gamma_%.2f", N, J, V, gamma) # to store
        collect_data_BH(
            N, 
            J,
            V, 
            gamma, 
            optimal, 
            t_end; 
            dissipation=dissipation,
            V2=V2,
            tau=tau,
            dt=dt,
            pre=pre,
            d_tracks=d_tracks,
            d_controls=d_controls,
            verbose=1,
            append=append
        )
    end
    
end

# set main() as standard function to run if not specified otherwise
if abspath(PROGRAM_FILE) == @__FILE__

    @time @everywhere push!(LOAD_PATH,"OptimalTrajectorySampler")
    @time include("ParallelTrajectorySampler.jl")

    @time main(ARGS)
end
    







