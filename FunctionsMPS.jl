function entropy_von_neumann!(psi::MPS, b::Int64)::Float64
"""Return VN entanglement entropy across bond b (i.e bipartite entanglement with separation A/B across bond b)"""
    
    # check if b is one of the edges, in that case return 0.
    if b < 1 || b >= length(psi)
        return 0.
    end
    
    # get site index
    s = siteinds(psi)
    
    # orthogonalize at bond b
    orthogonalize!(psi, b)
    
    # singular values
    _,S = svd(psi[b], (linkind(psi, b-1), s[b]))
    
    # get VN entanglement of those
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = S[n,n]^2 + 1E-7 # regularization to avoid divergences or Nan of sv is zero
        SvN -= p * log(p)
    end

    return SvN

end

function entropy_profile(psi::MPS)::Vector{Float64}
"""Compute full profile of entanglement of state psi, return as 1D Array of size L=length(psi)+1 (edges are zeroautomatically)"""

    N = length(psi)
    S = Float64[]
    
    # loop over all bonds, including boundaries
    for i in 0:N
        push!(S, entropy_von_neumann!(psi,i))
    end
    return S
end

function CdC(C::ITensor)::MPO
"""Compute C^daggerC for tensor C"""

    Cd = replaceprime(dag(C), 0 => 2 )
    return replaceprime(Cd*C, 2 => 1 )
end
