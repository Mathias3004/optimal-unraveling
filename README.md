# optimal-unraveling
Optimize stochastic quantum trajectories by minimizing trajectory entanglement entropy every jump click, using the MPS package itensor. 

## Source code

### OptimalTrajectorySampler
Julia source code for the optimal trajectory sampler. Each time step it is checked per interval of two sites if it clicks or not -- random offset for intervals. If it clicks, a 2x2 unitary optimization is applied to minimize the entanglement across the bond before the target interval.

### ParallelTrajectorySampler.jl
Function that runs a number of trajectories, with same parameters (i.e. Hamiltonian, dissipation, control parameters such dt, MBD etc), in parallel. If not specified otherwise (with n_samples), the number of available threads is used as default.

## Run code for XXZ model with uniform dephasing

### run_XXZ_trajectories.jl
Source file for running the parallel trajectories for XXZ Hamiltonian with dephasing $\sigma_z$. It runs trajectories in parallel on the number of threads available and stores results for entanglement profiles, sz-sz correlators and max bond dimensions in .txt files. One file contains a time slice of one quantity, with all trajectories lidsted as rows (in consistent order).

Parameters to specify: 
- N: system size
- gamma: dephasing rate
- Jz: sz-sz coupling in Hamiltonian (Jx=-1: set to unity ferromagnetic case)
- h: magnetic field
- optimization: 'none' or 'local' whether to apply the 2x2 optimization (local) or not (none)
- dir: directory to store the results

First, make folder to store: `mkdir data`.
Then, from command line: 
```
julia -p 128 --sysimage /jet/home/mathvr/.julia/sysimages/sys_itensors.so run_XXZ_trajectories.jl 20 1. 0. 0. local data
```

OR open REPL `julia -p 128 --sysimage /jet/home/mathvr/.julia/sysimages/sys_itensors.so` and run:
```
include("initialize_REPL.jl")
main(["20", "1.", "0.", "0.", "local", "data"])
```

This runs 128 parallel trajectories for t=[0,500] of size N=20, gamma=1., Jz=h=0. with local 2x2 optimization and stores results as .txt files in data/ (make sure dir 'data' exists), using precompiled itensor module.

### single_submit, loop_submit
bash scripts to submit a single slurm job (single_submit) or a series of jobs (loop_submit) by calling singe_submit for various parameters.

## Data analysis
Once the data have been collected, they are read in and analyzed with a number python notebooks

### generate_dataset.ipynb
Collect all the text files in a directory for some quantity (e.g entropy S) and save as .pickle for later analysis.

### analyze_dataset.ipynb
Read in the generated .pickle file and perforn analysis of obtained results by averaging over the obtained trajectories and make some plots of the scaling.


