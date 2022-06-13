@everywhere begin

    using ITensors
    using Optim
    using Distributed
    using DelimitedFiles
    
    # includes
    include("TEData.jl")
    include("Functions.jl")
    
    
    function dict_invert(d_in::Array{Dict{String,Any},1})::Dict{String,Any}
        """convert a 1D array of Dicts to a Dict of array. I.e. bring the key index outward"""
        ks = keys(d_in[1])
        N = length(d_in)
        
        # initialize
        d_return = Dict(name =>[] for name in ks)
        
        for d_sample in d_in
            for key in ks
                push!(d_return[key], d_sample[key] )
            end
        end
        
        #d_return = Dict(name => hcat(d_return[name]) for name in ks)
        
        return d_return
    end
    
    function dump_data(d_collect::Dict{String,Any}, prefix::String; write_append::String="a")
        """Save collected data d_collect to .txt file starting with prefix. write_append sets whether it appends or not ("a" or "w")."""
        ks = keys(d_collect)
        
        for key in ks
            filename = prefix * "_" * key * ".txt"
            data = transpose(d_collect[key])
            open(filename, write_append) do io
                writedlm(io, data)
            end
        end  
    end
        
    
    
    function collect_jump_probabilities(
            psi::MPS, 
            cdc_gates::Array{ITensor,1}
            )::Array{Float64,1}
        """Collect click probabilities. psi: actual state, cdc_gates: c^\dagger_i c_i of dissipators."""
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
        """Find average value of entropy in psi of applying one of two clicks ct1 or ct2 across bond i"""
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
        """Given two jumps c1 and c2, slect which one to apply to psi"""
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
        """two operators with inds i and i+1, determine optimal rotation (unitary) matrix to mix 
        c_i and c_{i+1} to minimize average entanglement after click using gradient descent"""
        s1 = ted.s[i]
        s2 = ted.s[i+1]
        c1 = ted.c_gates[i]*op("Id",s2)
        c2 = op("Id",s1)*ted.c_gates[i+1]
        
        orthogonalize!(psi,i)
        
        # local function to optimize wsith gradient descent
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
        """Sample which jumps click and apply them or perform non-Hermitian evolution"""
        pc = ted.dt*collect_jump_probabilities(psi, ted.cdc_gates)
         
        if optimal
            # random offset of mixing 2x2
            offset = mod( rand(Int64), 2 )
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
            iter = mod(rand(Int64), 2)==0 ? ((offset+1):2:(L-1)) : reverse((offset+1):2:(L-1))
            for i in iter
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
            optimal::Bool=false) # run optimal trajectory)
        """Evaluate a differential time step on psi. Both unitary and dissipative evolution"""
        
        # time step for numerical integration
        dt = ted.dt
        
        t_evolve = 0.
        while t_evolve < tau - dt/2.
            # apply Hamiltonian evolution
            psi = apply(ted.H_gates, psi; ted.cutoff, ted.maxdim)
            normalize!(psi)
            
            # select and apply jumps
            psi = select_and_apply_jumps!(ted, psi; optimal)
            t_evolve += dt
        end
        
        return psi
    end

end


function sample_trajectory(
    ted::TE_data,
    psi::MPS,
    t_end::Float64,
    tau::Float64,
    optimal::Bool;
    d_tracks::Dict{String, Any}=Nothing,
    verbose::Int64=1,
    parallel_step::Bool=true,
    n_samples::Int64=-1,
    pre::String="dat_",
    append::Bool=false
    )::Dict{String,Any}
    """
    Sample a full trajectoryusing parallel threads. 
    
    ted: contains Hamiltonian gates, dissipator gates and differential time step
    psi: initial state
    t_end: final time of evolution
    tau: time step to collect data
    optimal: whether to optimize the jump mixing or do bare
    d_tracks: dictionary to specify which quantities to track (entropy, list of observables, maxdim)
    verbose: verbose output or not
    parallel_step: whether to synchronize threads after each time step tau before writing result (recommended)
    n_samples: number of MC trajectory samples, default set to number of threads available
    pre: prefix .txt file to save data
    append: whether to append results to previously obtained samples (already existing text file). Default is overwrite
    returns Dict with all collected data
    """
    
    # function to calculta one time step and collect results
    function step_and_collect(psi::MPS, result_list::Array{Dict{String,Any},1})::MPS
        psi = sample_time_step(psi, ted, tau; 
            optimal=optimal)
        result = collect_properties(psi; d_tracks)
         #println("test")  
        push!(result_list, result)
        return psi
    end
    
    # set default value n_sample to number of threads
    if n_samples < 0
        n_samples = Threads.nthreads()
    end
    
    # number of time steps
    n_steps = Int64(ceil(t_end / tau))
    
    # time evolution and sample 
    if verbose > 0
        println("start time evolution...")
    end

    
    if !parallel_step
    
        function get_trajectory(dummy::Int64)::Dict{String,Any}
            result_list = Dict{String,Any}[]
            #println(psi)
            # first collection
            result = collect_properties(psi; d_tracks)
            push!(result_list, result)
            
            # initial state
            for i in 1:n_steps
                psi = step_and_collect(psi, result_list)
                
                # output
                if verbose > 0
                    println("Step $(i)/$(n_steps) obtained")
                end
            end
            
            # output
            if verbose > 0
                println("done")
            end
            #println(result_list)
            return dict_invert(result_list)
        end
        #println("test1")
        #sample_results = pmap(n -> get_trajectory(psi, d_tracks, n), 1:n_samples)
        sample_results = pmap(n -> get_trajectory(n), 1:n_samples)
        #println(n_samples)
        #println(sample_results)
        
    else
        #v_psi = SharedVector{MPS}(n_samples)
        #sample_results = SharedVector{Dict{String,Any}[]}(n_samples)
        @everywhere begin
            psi_l = $(psi)
            d_tracks_l = $(d_tracks)
            step_and_collect = $(step_and_collect)
            pre = $(pre)
            dat_coll = Dict{String,Any}[]
            push!(dat_coll, collect_properties(psi_l; d_tracks=d_tracks_l))
        end

        for i in 1:n_steps
            @sync @everywhere begin
                psi_l = step_and_collect(psi_l, dat_coll)
                prefix = pre * "_it_" * string($(i))
            end
            
            # save data from time step, keep order of processes
            for ip in 1:n_samples
                @everywhere if $(ip) == myid()
                    if $(ip)==1 && !$(append)
                        dump_data(dat_coll[end], prefix, write_append="w")
                    else
                        dump_data(dat_coll[end], prefix, write_append="a")
                    end
                end
            end

            if verbose > 0
                println("Step $(i)/$(n_steps) obtained")
            end
        end
        if verbose > 0
            println("done")
        end
        @everywhere dat_coll = dict_invert(dat_coll)
        
        sample_results = Dict{String,Any}[]
        for w in workers()
            push!(sample_results, @fetchfrom w dat_coll)
        end
        
    end
    
    return dict_invert(sample_results)
end




        

        
        
        
        
        
        
        

