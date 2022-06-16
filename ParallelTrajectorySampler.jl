# set number of workers to number of available threads
 @everywhere begin
    using TrajectorySampler
    using ITensors
end

function collect_trajectories_synchronized(
    ted::TE_data,
    psi::MPS,
    t_end::Float64,
    tau::Float64,
    optimal::Bool;
    d_tracks::Dict{String, Any}=Nothing,
    verbose::Int64=1,
    n_samples::Int64=-1,
    pre::String="dat_",
    append::Bool=false
    )
    """
    Sample a full trajectories using the number of parallel threads available and sync threads after each time step to collect the data. 
    
    ted: contains Hamiltonian gates, dissipator gates and differential time step
    psi: initial state
    t_end: final time of evolution
    tau: time step to collect data
    optimal: whether to optimize the jump mixing or do bare
    d_tracks: dictionary to specify which quantities to track (entropy, list of observables, maxdim)
    verbose: verbose output or not
    n_samples: number of MC trajectory samples, default set to number of threads available (keep it that way, otherwise you compute nthreads samples and only store n_sample < nthreads)
    pre: prefix .txt file to save data
    append: whether to append results to previously obtained samples (already existing text file). Default is overwrite
    """
    
    # local function to calculte one time step and collect results, defined on all threads
    function step_and_collect(psi::MPS)::Tuple{MPS,Dict{String,Any}}
        psi = sample_time_step(psi, ted, tau; 
            optimal=optimal)
        result = collect_properties(psi; d_tracks)
        return psi, result
    end
    
        # set default value n_sample to number of threads
    if n_samples < 0
        n_samples = nworkers()
    end
    
    # copy data on all threads
    @sync @everywhere begin
        psi_l = $(psi)
        d_tracks_l = $(d_tracks)
        step_and_collect = $(step_and_collect)
        #optimal = $(optimal)
        #d_tracks = $(d_tracks)
        pre = $(pre)
        #dat_coll = collect_properties(psi_l; d_tracks=d_tracks_l)
    end
        
    # number of time steps
    n_steps = Int64(ceil(t_end / tau))
    
    # time evolution and sample 
    if verbose > 0
        println("start time evolution of $(n_steps) steps to collect $(n_samples) samples using $(nworkers()) workers...")
    end

    # loop over all time steps to collect
    for i in 1:n_steps
        
        # update psi on all threads and collect data, prepare prefix for data saving
        @sync @everywhere begin
            psi_l, dat_coll = step_and_collect(psi_l)
            prefix = pre * "_it_" * string($(i))
            println("done")
        end
        
        # save data from time step, keep order of processes
        for ip in 1:n_samples
            @everywhere if $(ip) == myid()
                if $(ip)==1 && !$(append)
                    dump_data(dat_coll, prefix, write_append="w")
                else
                    dump_data(dat_coll, prefix, write_append="a")
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

end