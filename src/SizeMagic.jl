module SizeMagic

include("names.jl")
export named, nameless, ᵅ

# include("axes.jl")
export labelled, delabel, ᵡ


##### Borrowed #####

# see if you can compact these? and think about repeats?
# see also _dim, permutedims

@generated function order_named_inds(val::Val{L}, ni::NamedTuple{K}) where {L,K}
    tuple_issubset(K, L) || throw(DimensionMismatch("Expected subset of $L, got $K"))
    exs = map(L) do n
        if Base.sym_in(n, K)
            qn = QuoteNode(n)
            :(getfield(ni, $qn))
        else
            :(Colon())
        end
    end
    return Expr(:tuple, exs...)
end

Base.@pure function tuple_issubset(lhs::Tuple{Vararg{Symbol,N}}, rhs::Tuple{Vararg{Symbol,M}}) where {N,M}
    N <= M || return false
    for a in lhs
        found = false
        for b in rhs
            found |= a === b
        end
        found || return false
    end
    return true
end

end # module
