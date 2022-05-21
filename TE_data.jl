using ITensors

struct TE_data

    s::Vector{<:Index}

    H_gates::Array{ITensor,1} # hamiltonian gates
    c_gates::Array{ITensor,1} # jump gates (local)
    cdc_gates::Array{ITensor,1} # cd*c gates, precomputed
    Hcdc_gates::Array{ITensor,1} # exp( -0.5*cd*c*dt) gates, precomputed
    
    dt::Float64 # numerical discrete timestep used
    
    cutoff::Float64 # minimal sv
    maxdim::Int64 # maximal bond dim
end

struct XXZ_data
    N::Int64
    Jx::Float64
    Jz::Float64
    h::Float64
    gamma::Float64
    dissipation::String
end
    
    

function TE_data_XXZ(params::XXZ_data, dt::Float64; cutoff::Float64=1E-8, maxdim::Int64=300,conserve_qns::Bool=true)::TE_data

    s = siteinds("S=1/2", params.N; conserve_qns=conserve_qns)
    
    H_gates = get_H_XXZ(params.Jx, params.Jz, params.h, s, dt)
    c_gates, cdc_gates, Hcdc_gates = get_c(params.gamma, params.dissipation, s, dt)
    
    return TE_data(s, H_gates, c_gates, cdc_gates, Hcdc_gates, dt, cutoff, maxdim)
end

# prepare H gates
function get_H_XXZ(Jx::Float64, Jz::Float64, h::Float64, s::Vector{<:Index}, dt::Float64)::Array{ITensor,1}
    
    N = length(s)
    H_gates = ITensor[]

    # spin flip
    for j in 1:(N-1)
        s1 = s[j]
        s2 = s[j + 1]
        hj = 
        Jz*op("Sz", s1) * op("Sz", s2) +
             0.5*Jx *  (op("S+", s1) * op("S-", s2) + op("S-", s1) * op("S+", s2) )
        #hj =
      #op("Sz", s1) * op("Sz", s2) +
      #1 / 2 * op("S+", s1) * op("S-", s2) +
      #1 / 2 * op("S-", s1) * op("S+", s2)

        Gj = exp(-im * dt/2. * hj)
        push!(H_gates, Gj)
    end
    
    append!(H_gates, reverse(H_gates))
    
    # zeeman split
    if abs(h) > 1E-4
        for s1 in s
            hj = -h * op( "Sx", s1)
            Gj = exp(-im * dt * hj)
            push!(H_gates, Gj)
            
        end
    end
    return H_gates
end

function get_c(gamma::Float64, dissipation::String, s::Vector{<:Index}, dt::Float64)
    N = length(s)
    
    c_gates = ITensor[]
    cdc_gates = ITensor[]
    Hcdc_gates = ITensor[]
    for j in 1:N
        c = sqrt(gamma) * op(dissipation, s[j])
        cd = replaceprime(dag(c), 0 => 2 )
        cdc = replaceprime(cd*c, 2 => 1 )
        Hcdc = exp( - 0.5*cdc*dt)
        
        push!(c_gates, c)
        push!(cdc_gates, cdc)
        push!(Hcdc_gates, Hcdc)
    end
    return c_gates, cdc_gates, Hcdc_gates
end








    