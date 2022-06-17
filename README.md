# optimal-unraveling
Optimize stochastic quantum trajectories (minimal entanglement) by applying unitary transformation of jump operators

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
