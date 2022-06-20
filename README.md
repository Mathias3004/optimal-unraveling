## optimal-unraveling
Optimize stochastic quantum trajectories (minimal entanglement) by applying unitary transformation of jump operators
# Run 
* cd to directory and make dir to store data
```
mkdir store_XXZ
```

* Open Julia REPL with number of threads you want and have available
```
julia -p nthreads
```
* Add local path to all threads
```
@everywhere push!(LOAD_PATH,".")
```
* Load test for XXZ with dephasing for generating and storing data
```
include("run_XXZ_trajectories.jl")
```
* Run main function
```
main()
```
* Data are now stored in store_XXZ/ folder and you can use the template notebook Analyze_data.ipynb to read them and generate the plot.
# File description

* TrajectorySampler.jl: Module with all functions to sample a single trajectory with the possibility to sample clicks optimized over 2x2 unitary jump mixing to minimize average entanglement across bond after click (set by control parameter ```optimal```). The struct ```TE_data``` contains all the information related to Hamiltonian and dissipators to to evolve in time. Export function ```step_and_collect``` samples one sample time step ```tau``` and returns the observables as specified in ```d_tracks``` in dictionary, along with updated MPS.
* ParallelTrajectorySampler.jl: function ```collect_trajectories_synchronized``` creates parallel threads to sample trajectories. Each sample time step is run in parallel after wich threads are synchronized to collect results. Default is to employ all avalailable workers. The results are dumped in text files as specified by prefix ```pre```, one text file per variable per time step that contains the results of all parallel threads.
* run_XXZ_trajectories.jl: script to run an example for XXZ Hamiltonian with dephasing, containing functions ```get_H_XXZ``` and ```get_c``` to generate Hamiltonian and dissipator gates, respectively. ```collect_data_XXZ``` can be run to collect data for given parameter configuration (Jx,Jz,gamma etc). In ```main()``` is an example of a trajectory, with collection loop over the number of workers, for optimal and local (non-optimized) unravelings.
* Analyze_data.ipynb: read in all the collected text file and plot result for entropy in optimal and standard scenario to compare.
