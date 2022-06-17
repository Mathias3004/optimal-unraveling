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
    append: whether to append results to previously obtained samples (already existing text files). Default is overwrite
    """
    
    # set default value n_sample to number of workers
    if n_samples < 0
        n_samples = nworkers()
    end
    
    # copy relevant data to all workers
    @sync @everywhere begin
        psi_l = $(psi)
        ted = $(ted)
        tau = $(tau)
        optimal = $(optimal)
        d_tracks = $(d_tracks)
        pre = $(pre)
    end
        
    # number of time steps
    n_steps = Int64(ceil(t_end / tau))
    
    # time evolution and sample 
    if verbose > 0
        println("\nStart time evolution of $(n_steps) steps to collect $(n_samples) samples using $(nworkers()) workers")
    end

    # loop over all time steps to collect
    for i in 1:n_steps
        
        # update psi on all threads and collect data, prepare prefix for data saving
        if verbose > 0
            print("\nStep $(i)/$(n_steps)...\t")
        end
        @sync @everywhere begin
            psi_l, dat_coll = step_and_collect(psi_l, ted, tau, optimal, d_tracks)
            prefix = pre * "_it_" * string($(i))
            if $verbose > 1
                println("step evaluated")
            end
        end
        
        # save data from time step, keep order of processes
        if verbose > 0
            print("dumping data...\t")
        end
        for ip in workers()
            # check whether to append or not (only first worker worker, the rest is always mode)
            if ip==workers()[1] && !append
                write_append = "w"
            else
                write_append = "a"
            end
            
            @everywhere if $(ip) == myid()
                dump_data(dat_coll, prefix, write_append=$(write_append) )
            end
        end

        if verbose > 0
            print("done")
        end
    end
    if verbose > 0
        println("\n\nAll steps obtained!\n")
    end

end