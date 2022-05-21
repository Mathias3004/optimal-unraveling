using ITensors
using Optim

function collect_jump_probabilities(
        psi::MPS, 
        cdc_gates::Array{ITensor,1})::Array{Float64,1}
    # collect click probabilities
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

function get_average_click_entropy(psi::MPS, ct1::ITensor, ct2::ITensor, i::Int64)::Float64
    # apply jumps
    psi1 = apply(ct1, psi)
    psi2 = apply(ct2, psi)
    
    # get probabilities jumps
    p1 = norm(psi1)^2
    p2 = norm(psi2)^2
    
    # evaluate entanglement outcomes
    orthogonalize!(psi1,i)
    orthogonalize!(psi2,i)
    return p1*entropy_von_neumann!(psi1, i) + p2*entropy_von_neumann!(psi2, i)
end

function select_and_apply(psi::MPS, c1::ITensor, c2::ITensor)::MPS
    # apply jumps
    psi1 = apply(c1, psi)
    psi2 = apply(c2, psi)
    
    # get probabilities jumps
    p1 = norm(psi1)^2
    p2 = norm(psi2)^2
    
    if p1 > (p1+p2)*rand(Float64)
        return normalize!(psi1)
    else
        return normalize!(psi2)
    end
end

function apply_optimal_jump(psi::MPS, ted::TE_data, i::Int64)::MPS
    s1 = ted.s[i]
    s2 = ted.s[i+1]
    c1 = ted.c_gates[i]*op("Id",s2)
    c2 = op("Id",s1)*ted.c_gates[i+1]
    
    orthogonalize!(psi,i)
    
    # local function to optimize
    function varS(pars::Vector{Float64})
        theta = pars[1]
        phi = pars[2]
        ct1 = cos(theta)*c1 + exp(im*phi)*sin(theta)*c2
        ct2 = -exp(-im*phi)*sin(theta)*c1 + cos(theta)*c2
        return get_average_click_entropy(psi,ct1,ct2,i)
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
        optimal::Bool=false)
    
    pc = ted.dt*collect_jump_probabilities(psi, ted.cdc_gates)
     
    if optimal
        # random offset of mixing 2x2
        offset = mod( rand(Int), 2 )
        L = length(psi)
        
        # check boundaries and apply jumps
        if offset == 1 && pc[1] > rand(Float64)
            psi = apply(ted.c_gates[1], psi)
        end
        if mod(L-offset,2) == 1 && pc[end] > rand(Float64)
            psi = apply(ted.c_gates[end], psi)
        end
        normalize!(psi)
        
        # loop through chain, per two sites, starting from offset (c_i and c_{i+1} will be mixed)
        for i in (offset+1):2:(L-1)
            orthogonalize!(psi,i)
            # two jumps
            if pc[i]*pc[i+1] > rand(Float64)
                psi = apply(ted.c_gates[i:i+1], psi)
            # one jump
            elseif pc[i] + pc[i+1] > rand(Float64)
                psi = apply_optimal_jump(psi, ted, i)
            # no jump
            else
                psi = apply(ted.Hcdc_gates[i:i+1],psi)
            end
            
            normalize!(psi)
        end
        
    println(pc[1])   
    else # just do direct jumps
        for (i,p) in enumerate(pc)
            if p > rand(Float64)
                psi = apply(ted.c_gates[i], psi; cutoff=ted.cutoff, maxdim=ted.maxdim)
            else
                psi = apply(ted.Hcdc_gates[i], psi; cutoff=ted.cutoff, maxdim=ted.maxdim)  
            end
            normalize!(psi)
        end
    end
    return psi
        
end

function collect_data(psi::MPS, track_entropy::Bool, track_observables::Vector{String})
    data_t = Dict()
    
    if track_entropy
        data_t["S"] = entropy_profile(psi)
    end
    
    if length(track_observables) != 0
        for obs in track_observables
            data_t[obs] = expect(psi, obs)
        end
    end
    
    return data_t
end
            
            
        
    
function sample_trajectory(
        psi::MPS,
        ted::TE_data,
        v_t::Array{Float64,1};
        optimal::Bool=false,
        track_states::Bool=false, 
        track_entropy::Bool=true, 
        track_local_observables::Vector{String}=String[])
    
    # time step for data collection
    tau = v_t[2]-v_t[1]
    
    # time step for numerical integration
    dt = ted.dt
    
    # final time
    t_end = maximum(v_t)
    
    # initialize to store data
    data = Dict()
    if track_states
        data["psi"] = MPS[]
    end
    if track_entropy
         data["S"] = zeros((length(v_t),length(psi)+1))
    end
    if length(track_local_observables) != 0
        for obs in track_local_observables
            data[obs] = zeros( (length(v_t),length(psi)) )
        end
    end
    
    # collect data initial states
    if track_states
        push!(data["psi"], psi)
    end
    data_t = collect_data(psi, track_entropy, track_observables)
    for obs in keys(data_t)
        data[obs][1,:] = data_t[obs]
    end
    
    t_evolve = 0.
    t_next = v_t[2]
    ind_collect = 2
    while t_evolve < t_end-dt/2.
        # apply Hamiltonian evolution
        psi = apply(ted.H_gates, psi; ted.cutoff, ted.maxdim)
        normalize!(psi)
        
        # select and apply jumps
        psi = select_and_apply_jumps!(ted, psi; optimal)
        t_evolve += dt
        
        if t_evolve >= t_next-1E-4
        
            # store data in dictionary
            if track_states
                push!(data["psi"], psi)
            end
            data_t = collect_data(psi, track_entropy, track_observables)
            for obs in keys(data_t)
                data[obs][ind_collect,:] = data_t[obs]
            end
            
            # for next data collection
            ind_collect += 1
            if !(ind_collect > length(v_t)  )
                t_next = v_t[ind_collect]
            end
        end
        
        
        
    end
    
    return data
end




