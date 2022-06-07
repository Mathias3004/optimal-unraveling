    
    # collect data initial states
    if collect_init
        if track_states
            push!(data["psi"], psi)
        end
        if track_maxdim
            data["maxdim"][1] = maxlinkdim(psi)
        end
        data_t = collect_data(psi, track_entropy, track_local_observables)
        for obs in keys(data_t)
            data[obs][1,:] = data_t[obs]
        end
    end
    
    function sample_trajectory(
        psi::MPS, # state coming in
        ted::TE_data, # data for time evolution
        tau::Float64; # time to integrate
        optimal::Bool=false, # run optimal trajectory
        track_states::Bool=false,
        track_maxdim::Bool=true,
        track_entropy::Bool=true, 
        track_local_observables::Vector{String}=String[])
    
    # time step for numerical integration
    dt = ted.dt
    
    t_evolve = 0.
    t_next = v_t[2]
    ind_collect = 2
    
    if verbose > 0
        println("start time evolution...")
    end
    
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
            if track_maxdim
                data["maxdim"][ind_collect] = maxlinkdim(psi)
            end
             
            data_t = collect_data(psi, track_entropy, track_local_observables)
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

        if t_evolve >= t_next-1E-4
        
            # store data in dictionary
            if track_states
                push!(data["psi"], psi)
            end
            if track_maxdim
                data["maxdim"][ind_collect] = maxlinkdim(psi)
            end
             
            data_t = collect_data(psi, track_entropy, track_local_observables)
            for obs in keys(data_t)
                data[obs][ind_collect,:] = data_t[obs]
            end
            
            # for next data collection
            ind_collect += 1
            if !(ind_collect > length(v_t)  )
                t_next = v_t[ind_collect]
            end
        end
        
        
         # initialize to store data
    #= data = Dict{String,Array{Float64}}
    if track_states
        data["psi"] = psi
    end
    if track_maxdim
        data["maxdim"] = zeros(length(v_t))
    end
    if track_entropy
         data["S"] = zeros((length(v_t), length(psi)+1))
    end
    if length(track_local_observables) != 0
        for obs in track_local_observables
            data[obs] = zeros((length(v_t), length(psi)))
        end
    end =#
    

    #data_opt = Array{Any,1}
    #data_bare = Array{Any, 1} 

    ############################################################################
    # time evolution and sample 
    if verbose > 0
        println("start time evolution...")
    end
    
    result_list = Dict{String,Any}[]
    
    # first collection
    result = collect_properties(psi;
        track_states=track_states, 
        track_entropy=track_entropy, 
        track_local_observables=track_local_observables)
        
    push!(result_list, result)
    
    # number of time steps
    n_steps = Int64(ceil(t_end / tau))
    
    for i in 1:n_steps
        psi = sample_time_step(psi, ted, tau; 
            optimal=optimal)
        result = collect_properties(psi;
            track_states=track_states, 
            track_entropy=track_entropy, 
            track_local_observables=track_local_observables)
            
        push!(result_list, result)
        
        if verbose > 0
            println("Step $(i)/$(n_steps) obtained")
        end
    end
    data_return = dict_invert(result_list)   
    # parallel process and collect data
    
    print("Done!")