module OptimalTrajectorySampler

println("Precompiling OptimalTrajectorySampler...")

export TE_data, step_and_collect, dump_data

using ITensors 
using Optim
using DelimitedFiles
using Statistics

include("FunctionsData.jl") # functions to process/save data
include("FunctionsMPS.jl") # general MPS functions (such as entanglement)

struct TE_data
    """struct containing all information for evaluating time evolution"""

    s::Vector{<:Index}

    H_gates::Array{ITensor,1} # hamiltonian gates
    c_gates::Array{ITensor,1} # jump gates (local)
    cdc_gates::Array{ITensor,1} # cd*c gates, precomputed
    Hcdc_gates::Array{ITensor,1} # exp( -0.5*cd*c*dt) gates, precomputed
    
    dt::Float64 # numerical discrete timestep used
    
    cutoff::Float64 # minimal sv
    maxdim::Int64 # maximal bond dim
end

function step_and_collect(
    psi::MPS,
    ted::TE_data,
    tau::Float64, 
    optimal::String,
    d_tracks::Dict{String,Any}
    )::Tuple{MPS,Dict{String,Any}}
    """Exported function: evolve psi over time tau using ted, optimized or not, 
    and collect data as specified in d_tracks. Return new state psi and dict of results."""
    
    psi = sample_time_step(psi, ted, tau; 
        optimal=optimal)
    result = collect_properties(psi; d_tracks)
    return psi, result
end

function collect_jump_probabilities(
        psi::MPS, 
        cdc_gates::Array{ITensor,1}
        )::Array{Float64,1}
    """Collect click probabilities. psi: actual state, cdc_gates: c^dagger_i c_i of dissipators."""
    normalize(psi)
    pc = Float64[]
    for (i, cdc) in enumerate(cdc_gates)
        orthogonalize!(psi,i)
        ket = psi[i]
        bra = dag(prime(ket,"Site"))
        push!(pc, real((bra*cdc*ket)[1]) )
        #push!(pc, expect(psi,cdc))
    end
        
    return pc
end

function get_average_click_entropy_local(psi::MPS, ct1::ITensor, ct2::ITensor, i::Int64; jump_threshold::Float64=1e-3)::Float64
    """Find average value of entropy in psi of applying one of two clicks ct1 or ct2 across bond i"""
    
    # apply jumps to psi
    psi1 = apply(ct1, psi)
    psi2 = apply(ct2, psi)
    
    # get probabilities jumps
    p1 = norm(psi1)^2
    p2 = norm(psi2)^2
    
    # evaluate entanglement outcomes
    cost = 0.
    if p1 > jump_threshold
        cost += p1*entropy_von_neumann!(orthogonalize!(psi1,i), i)
    end
    
    if p2 > jump_threshold
        cost += p2*entropy_von_neumann!(orthogonalize!(psi2,i), i)
    end
    
    return cost
end

function get_average_click_entropy_global(psi::MPS,ct1::ITensor,ct2::ITensor)::Float64
    # apply jumps to psi
    psi1 = apply(ct1, psi)
    psi2 = apply(ct2, psi)
    
    # get probabilities jumps
    p1 = norm(psi1)^2
    p2 = norm(psi2)^2
    
    return p1*mean(entropy_profile(psi1)) + p2*mean(entropy_profile(psi2))
end
    

function select_and_apply(psi::MPS, c1::ITensor, c2::ITensor; jump_threshold::Float64=1e-3)::MPS
    """Given two jumps c1 and c2, select which one to apply to psi"""
    
    # apply jumps to psi
    psi1 = apply(c1, psi)
    psi2 = apply(c2, psi)
    
    # get probabilities jumps
    p1 = norm(psi1)^2
    p2 = norm(psi2)^2
    
    if p1 > (p1+p2)*max(rand(Float64), jump_threshold)
        return normalize!(psi1)
    else
        return normalize!(psi2)
    end
end

function apply_optimal_jump(psi::MPS, ted::TE_data, i::Int64, optimal::String; jump_threshold::Float64=1e-3)::MPS
    """two operators with inds i and i+1, determine optimal rotation (unitary) matrix to mix 
    c_i and c_{i+1} to minimize average entanglement after click using gradient descent"""
    
    # get sites to optimize over
    s1 = ted.s[i]
    s2 = ted.s[i+1]
    c1 = ted.c_gates[i]*op("Id",s2)
    c2 = op("Id",s1)*ted.c_gates[i+1]
    
    # orthogonalize psi at right bond
    orthogonalize!(psi,i)
    
    # local function to optimize wsith gradient descent
    function varS(pars::Vector{Float64})
        theta = pars[1]
        phi = pars[2]
        ct1 = cos(theta)*c1 + exp(im*phi)*sin(theta)*c2
        ct2 = -exp(-im*phi)*sin(theta)*c1 + cos(theta)*c2
        
        if optimal == "global"
            return get_average_click_entropy_global(psi,ct1,ct2)
        else
            return get_average_click_entropy_local(psi,ct1,ct2,i; jump_threshold)
        end
    end
    # optimization and find optimal theta, phi
    res = optimize(varS, [0.0, 0.0])
    pars_opt = Optim.minimizer(res)
    theta_opt, phi_opt = pars_opt[1], pars_opt[2]
    
    #optimal operators
    ct1_opt = cos(theta_opt)*c1 + exp(im*phi_opt)*sin(theta_opt)*c2
    ct2_opt = -exp(-im*phi_opt)*sin(theta_opt)*c1 + cos(theta_opt)*c2
    
    return select_and_apply(psi, ct1_opt, ct2_opt)

end

function select_and_apply_jumps!(
        ted::TE_data,
        psi::MPS;
        optimal::String="local",
        jump_threshold::Float64=1e-3)
    """Sample which jumps click and apply them or perform non-Hermitian evolution. 
    Option optimal specifies whether 2x2 optimal U are obtained to minimize entanglement across bond, otherwise direct jump clicks"""
    
    # collect probabilities for each site to click
    pc = ted.dt*collect_jump_probabilities(psi, ted.cdc_gates)
    
    
    if optimal != "none" # perform optimization for jumps 
        
        # random offset of mixing 2x2
        offset = mod(rand(Int64), 2)
        L = length(psi)
        
        # check boundaries and apply direct jumps at edges if required
        if offset == 1 && pc[1] > rand(Float64)
            psi = apply(ted.c_gates[1], psi)
        end
        if mod(L-offset,2) == 1 && pc[end] > rand(Float64)
            psi = apply(ted.c_gates[end], psi)
        end
        normalize!(psi)
        
        # loop through chain, per two sites, starting from offset (c_i and c_{i+1} will be mixed)
        iter = mod(rand(Int64), 2)==0 ? ((offset+1):2:(L-1)) : reverse((offset+1):2:(L-1))
        for i in iter
            orthogonalize!(psi,i)
            # two jumps (taken out, might cause numerical problems: probabilities calculated independently (linear) but here non-linear. 
            # First jump might obstruct second, resulting in nearly zero norm after application.)
            #if pc[i]*pc[i+1] > rand(Float64)
            #    psi = apply(ted.c_gates[i:i+1], psi)
            # one jump
            if pc[i] + pc[i+1] > max(rand(Float64), 2. *jump_threshold*ted.dt) # apply jump threshold to avoid ending up with nearly zero norm after applying jump
                psi = apply_optimal_jump(psi, ted, i, optimal; jump_threshold)
            # no jump
            else
                psi = apply(ted.Hcdc_gates[i:i+1],psi)
            end
            
            normalize!(psi)
        end
        
    else # just do direct jumps, no optimization
        last_p = 0.
        last_norm_prev = 0.
        last_norm = 0.
        for (i,p) in enumerate(pc)
            if p > max(rand(Float64), jump_threshold*ted.dt)
                last_norm_prev = norm(psi)
                psi = apply(ted.c_gates[i], psi; cutoff=ted.cutoff, maxdim=ted.maxdim)
                last_p = p
                last_norm = norm(psi)
            else
                try
                    psi = apply(ted.Hcdc_gates[i], psi; cutoff=ted.cutoff, maxdim=ted.maxdim)  
                catch e
                    println(i)
                    println("last norm psi is $(last_norm_prev)")
                    println("last norm psi is $(last_norm)")
                    println("last p is $last_p")
                    println(pc)
                    rethrow(e)
                end
                    
            end
            normalize!(psi)
        end
    end
    return psi
        
end

function collect_properties(
        psi;
        d_tracks::Dict{String,Any}=Nothing
        )::Dict{String,Any}
    """Evaluate the properties contained in d_tracks on state pse. Return as dictionary"""
    if d_tracks == Nothing
        d_tracks = Dict("track_states" => false, "track_maxdim" => true, "track_entropy" => true, "track_local_observables" => String[])
    end

    data = Dict()
    if d_tracks["track_states"]
        data["psi"] = psi
    end
    if d_tracks["track_maxdim"]
        data["maxdim"] = maxlinkdim(psi)
    end
    
    if d_tracks["track_entropy"]
        data["S"] = entropy_profile(psi)
    end
    
    if length(d_tracks["track_local_observables"]) != 0
        for obs in d_tracks["track_local_observables"]
            data[obs] = expect(psi, obs)
        end
    end
    
    return data
end
           
function sample_time_step(
        psi::MPS, # state coming in
        ted::TE_data, # data for time evolution
        tau::Float64; # time to integrate
        optimal::String="local" # run optimal trajectory
        )::MPS
    """Evaluate a differential time step on psi. Both unitary and dissipative evolution"""
    
    # time step for numerical integration
    dt = ted.dt
    
    t_evolve = 0.
    while t_evolve < tau - dt/2.
        # apply Hamiltonian evolution
        #norm = norm(psi)
        try
            psi = apply(ted.H_gates, psi; ted.cutoff, ted.maxdim)
            normalize!(psi)
        catch e
            
            println("norm psi is $(norm(psi))")
            rethrow(e)
        end
        
        # select and apply jumps
        psi = select_and_apply_jumps!(ted, psi; optimal)
        t_evolve += dt
    end
    
    return psi
end

# precompilations
precompile(step_and_collect, (MPS, TE_data, Float64, Bool, Dict{String,Any},)) 

end # module
