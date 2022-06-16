function dict_invert(d_in::Array{Dict{String,Any},1})::Dict{String,Any}
    """Convert a 1D array of Dicts to a Dict of arrays, with same keys. I.e. bring the key index outwards"""
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
        data = transpose(d_collect[key]) # transpose to write arrays as rows in txt file (so each row is a collected sample)
        open(filename, write_append) do io
            writedlm(io, data)
        end
    end  
end