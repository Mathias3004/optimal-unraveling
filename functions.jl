using ITensors

function entropy_von_neumann!(psi::MPS, b::Int64)
    s = siteinds(psi)  
    if b < 1 || b >= length(psi)
        return 0.
    end

    orthogonalize!(psi, b)
    _,S = svd(psi[b], (linkind(psi, b-1), s[b]))
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end

    return SvN

end

function entropy_profile(psi::MPS)::Vector{Float64}
    N = length(psi)
    S = Float64[]
    for i in 0:N
        push!(S, entropy_von_neumann!(psi,i))
    end
    return S
end


function CdC(C::ITensor)::MPO
    Cd = replaceprime(dag(C), 0 => 2 )
    return replaceprime(Cd*C, 2 => 1 )
end