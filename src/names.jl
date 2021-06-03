export NamedInt, NDA, _value, _name, _names, _dim # for now!

abstract type AbstractNI{L} <: Integer end

"""
    NamedInt{:μ}(3) :: AbstractNI

An `Int` with a name attached! Using these for `size` is the
key idea of this package. 

Most operations discard the name. Only `+` and `==` preserve it,
and give an error if two names (neither `_`) don't match.
"""
struct NamedInt{L} <: AbstractNI{L}
    val::Int
end
NamedInt{L}(x::NamedInt{L}) where {L} = x

_named(x::Integer, s::Symbol) = NamedInt{s}(x)
_named(x::Bool, ::Symbol) = x
_named(x, _) = x
_name(x::AbstractNI{L}) where {L} = L
_name(x::Integer) = :_
_value(x::NamedInt) = x.val
_value(x) = x

Base.show(io::IO, x::NamedInt{L}) where {L} = print(io, x.val, "ᵅ")
Base.show(io::IO, ::MIME"text/plain", x::NamedInt{L}) where {L} = print(io, "NamedInt{$L}(", x.val, ")")
const ᵅ = NamedInt{:_}(1) # allows you to paste what's printed

Base.promote_rule(::Type{<:AbstractNI}, ::Type{T}) where {T<:Number} = T
Base.promote_rule(::Type{<:AbstractNI}, ::Type{<:AbstractNI}) = Int
# Base.promote_rule(::Type{<:NamedInt{L}}, ::Type{<:NamedInt{L}}) where {L} = NamedInt{L}

Base.convert(::Type{T}, x::AbstractNI) where {T<:Number} = convert(T, _value(x))
(::Type{T})(x::AbstractNI) where {T<:Number} = T(_value(x))
(::Type{NamedInt{L}})(x::NamedInt{L2}) where {L,L2} = NamedInt{_combine(L,L2)}(x.val)

for op in (:-, :*, :&, :|, :xor, :<, :<=)
    @eval Base.$op(a::AbstractNI, b::AbstractNI) = $op(_value(a), _value(b))
end

for f in (:abs, :abs2, :sign, :-)
    @eval Base.$f(x::AbstractNI) = $f(_value(x))
end

for op in (:+, :(==))
    @eval Base.$op(a::NamedInt, b::NamedInt) = _apply($op, a, b)
    @eval Base.$op(a::NamedInt, b::Integer) = _apply($op, a, b)
    @eval Base.$op(a::Integer, b::NamedInt) = _apply($op, a, b)
end
# _apply(f, a, b) = begin _combine(_name(a), _name(b)); f(_value(a), _value(b)) end
_apply(f, a, b) = _named(f(_value(a), _value(b)), _combine(_name(a), _name(b), true))
_combine(x::Symbol, y::Symbol, strict::Bool=true) = x === :_ ? y : y === :_ ? x : x === y ? x :
    strict ? throw("name mismatch 😧 :$x != :$y") : :_
# Surely Base._cs etc could share code here?

Base.sum(t::Tuple{AbstractNI, Vararg{Integer}}) = sum(_value, t) # needed for show, within print_matrix

abstract type AbstractNDA{L,T,N} <: AbstractArray{T,N} end

"""
    NDA(array, names) :: AbstractNDA

Wrapper which attaches names, a tuple of symbols, to an array. 
The goal is to work like NamedDimsArray, with far less code,
by adding methods to lower-level functions to pass `NamedInt`s along.
"""
struct NDA{L,T,N,P} <: AbstractNDA{L,T,N}
    data::P
    NDA(data::P, names::NTuple{N,Symbol}=ntuple(d->:_, N)) where {P<:AbstractArray{T,N}} where {T,N} =
        new{names,T,N,P}(data)
end

Base.size(A::NDA{L}) where L = map((ℓ,s) -> NamedInt{ℓ}(s), L, size(A.data))
Base.axes(A::NDA{L}) where L = map(_nameaxis, L, axes(A.data))
_nameaxis(l::Symbol, ax) = ax
_nameaxis(l::Symbol, ax::Base.OneTo) = Base.OneTo(NamedInt{l}(ax.stop))

Base.getindex(A::AbstractNDA{L,T,N}, I::Vararg{Int,N}) where {L,T,N} = getindex(A.data, I...)
Base.setindex!(A::AbstractNDA{L,T,N}, val, I::Vararg{Int,N}) where {L,T,N} = setindex!(A.data, val, I...)

# Not sure these get called
# Base.to_index(i::NamedInt) = Base.to_index(_value(i))
# Base.to_indices(A, inds, I::Tuple{NamedInt, Vararg{Any}}) = (_value(I[1]), Base.to_indices(A, Base.tail(inds), Base.tail(I))...)

# Base.strides(A::NDA) = strides(A.data)
# Base.stride(A::NDA, d::Int) = stride(A.data, d)
# Base.unsafe_convert(::Type{Ptr{T}}, A::NDA) where {T} = unsafe_convert(Ptr{T}, A.data)
# Base.elsize(::Type{<:NDA{L,T,N,P}}) where {L,T,N,P} = Base.elsize(P)

_value(A::NDA) = _value(A.data) # keep looking inside!
_names(A::AbstractArray) = map(_name, size(A)) # this way it works on wrappers

for fun in (:getindex, :view, :dotview, :setindex!)  # Keyword indexing, PIRACY!!
    extra = fun == :setindex! ? (:val,) : ()
    @eval Base.$fun(A::AbstractArray, $(extra...); kw...) = 
        Base.$fun(A, $(extra...), ntuple(d -> get(kw, _names(A)[d], :), ndims(A))...) # matches repeated names, and _
        # Base.$fun(A, $(extra...), order_named_inds(Val(_names(A)), kw.data)...)
end

###### Construction

Base.similar(A::AbstractArray, ::Type{T}, dims::Tuple{NamedInt, Vararg{Integer}}) where {T} =
    NDA(similar(A, T, map(_value, dims)), map(_name, dims))
Base.similar(::Type{A}, dims::Tuple{NamedInt, Vararg{Integer}}) where {A<:AbstractArray} =
    NDA(similar(A, map(_value, dims)), map(_name, dims))

(::Type{A})(::UndefInitializer, sz) where {A<:NDA{L,T,N,P}} where {L,T,N,P} = NDA(P(undef, sz), L)

# (Array{T})(::UndefInitializer, a::NamedInt, bs::Integer...) where {T} = 
#     NDA(Array{T}(undef, _value(a), map(_value, bs)...), (_name(a), map(name, bs)...)) # dodgy but makes @einsum work

function Base.reshape(A::AbstractArray, shp::Tuple{<:NamedInt, Vararg{Integer}})
    data = reshape(_value(A), map(_value, shp))
    NDA(data, map(_name, shp))  # todo -- check against previous, where sizes match? 
end
function Base._dropdims(A::AbstractNDA, dims::Base.Dims)
    ax = Base._foldoneto((ds, d) -> d in dims ? ds : (ds..., axes(A,d)), (), Val(ndims(A)))
    reshape(_value(A), ax)
end

###### ??

_NamedOneTo{L} = Base.OneTo{NamedInt{L}}
_name(r::_NamedOneTo{L}) where {L} = L
_value(r::_NamedOneTo) = Base.OneTo(_value(r.stop))

_named(r::Base.OneTo, s::Symbol) = Base.OneTo(_named(r.stop, s))

# Base.last(r::_NamedOneTo) = _value(r.stop) # for lastindex, means 2:end loses the name
# Base.UnitRange(r::_NamedOneTo) = UnitRange(_value(r)) # needed for show with last

Broadcast.axistype(a::_NamedOneTo, b::Base.OneTo) = a
Broadcast.axistype(a::Base.OneTo, b::_NamedOneTo{L}) where {L} = Base.OneTo(_named(a.stop, L))
Broadcast.axistype(a::_NamedOneTo{L}, b::_NamedOneTo{L2}) where {L,L2} = Base.OneTo(_named(a.stop, _combine(L,L2,true)))

# Base._cs(d, a::AbstractNI, b::AbstractNI) = _named(Base._cs(d, _value(a), _value(b)), _combine(_name(a), _name(b), true)) # used by cat(A, A'; dims=3), ==

# Base.promote_shape(ax::Tuple{_NamedOneTo,Vararg{<:Base.OneTo}}, bx::Base.Indices) = _promoteshape(ax, bx)
# _promoteshape(ax, bx) = 
# This is to make map(+, A, A') give an error, if == is not strict. May be desirably anyway, to promote? 

Base.to_shape(i::NamedInt) = i  # needed for hcat etc.
Base.to_shape(r::_NamedOneTo) = last(r)
_toshape(r::_NamedOneTo) = last(r)
_toshape(r::Base.OneTo) = NamedInt{:_}(last(r))
Base.to_shape(ax::Tuple{_NamedOneTo,Base.OneTo,Vararg{<:Base.OneTo}}) = map(_toshape, ax)
Base.to_shape(ax::Tuple{Base.OneTo,_NamedOneTo,Vararg{<:Base.OneTo}}) = map(_toshape, ax)
Base.to_shape(ax::Tuple{_NamedOneTo,_NamedOneTo,Vararg{<:Base.OneTo}}) = map(_toshape, ax)

Base.unsafe_length(r::_NamedOneTo{L}) where {L} = NamedInt{L}(r.stop - zero(r.stop)) # for view
Base.unsafe_length(r::AbstractUnitRange{NamedInt{L}}) where {L} = NamedInt{L}(last(r) - first(r) + step(r)) # a:b

_value(r::UnitRange{<:NamedInt}) = UnitRange(_value(r.start), _value(r.stop))
_value(v::SubArray) = SubArray(_value(v.parent), v.indices)
_value(A::PermutedDimsArray{T,N,P}) where {T,N,P} = PermutedDimsArray(_value(parent(A)), P)

# (c::Colon)(start::Integer, stop::NamedInt{L}) where {L} = NDA(c(start, stop.val), (L,)) # explicit wrap
(::Colon)(start::Integer, stop::NamedInt{L}) where {L} = UnitRange{NamedInt{L}}(NamedInt{L}(start), stop) # wants Base.unsafe_length

# Base._array_for(::Type{T}, itr::Base.Generator{_NamedOneTo}, ::Base.HasLength) where {T} = @show "yes"
# Base._array_for(::Type{T}, itr::Base.Generator{_NamedOneTo}, ::Base.HasShape{N}) where {T,N} = @show "yes"

##### Reductions & dims

Base._dropdims(A::AbstractArray, name::Symbol) = Base._dropdims(A, (_dim(A, name),)) # This is why sum() should keep names. Also, PIRACY!
Base._dropdims(A::AbstractArray, names::Tuple{Vararg{Symbol}}) = Base._dropdims(A, _dim(A, names)) # PIRACY!

# Base.reduced_indices(ax::Tuple{AbstractUnitRange{<:NamedInt}, Vararg{AbstractUnitRange}}, dim::Int) = ntuple(d -> d==dim ? Base.OneTo(1) : ax[d], length(ax)) # this seems too late, gives a stack overflow?

Base.reducedim_initarray(A::NDA{L}, region, init, ::Type{R}) where {L,R} =
    NDA(Base.reducedim_initarray(A.data, _dim(A,region), init, R), L)  # this makes sum(A, dims=1) work

Base.reducedim_initarray(A::AbstractArray, region::Union{Symbol, Tuple{Vararg{Symbol}}}, init, ::Type{R}) where {R} =
    Base.reducedim_initarray(A, _dim(A, region), init, R) # PIRACY, makes sum(A', dims=:x) work
Base.reducedim_initarray(A::NDA, region::Union{Symbol, Tuple{Vararg{Symbol}}}, init, ::Type{R}) where {R} =
    Base.reducedim_initarray(A, _dim(A, region), init, R) # solve ambiguity

_dim(A::AbstractArray, d) = d
_dim(A::AbstractArray, s::Symbol) = begin d = findfirst(n -> n===s, _names(A)); d===nothing && throw("name $d not found in $(_names(A))"); d end
_dim(A::AbstractArray, ds::Tuple{Vararg{Symbol}}) = map(s -> _dim(A,s), ds)

Base.eachslice(A::AbstractNDA; dims) = invoke(eachslice, Tuple{AbstractArray}, A; dims=_dim(A, dims))
Base.mapslices(f, A::AbstractNDA; dims) = invoke(mapslices, Tuple{Any, AbstractArray}, f, A; dims=_dim(A, dims))
function Base.cumsum(A::AbstractNDA{L,T}; dims) where {L,T}
    out = similar(A, Base.promote_op(Base.add_sum, T, T))
    cumsum!(out, A, dims=_dim(A,dims))
end
Base.cumprod(A::AbstractNDA; dims) = accumulate(Base.mul_prod, A, dims=_dim(A, dims))

Base.PermutedDimsArrays.genperm(ax::Tuple{_NamedOneTo, Vararg{Any}}, perm::Tuple{Symbol, Vararg{Any}}) = begin
    axnames = map(_name, ax)
    map(s -> ax[findfirst(isequal(s), axnames)], perm)
end
Base.permutedims!(dest, src::AbstractArray, perm::Tuple{Vararg{Symbol}}) = permutedims!(dest, src, map(s -> findfirst(isequal(s), _names(src)), perm))  # PIRACY!
Base.PermutedDimsArray(src::AbstractArray, perm::Tuple{Vararg{Symbol}}) = PermutedDimsArray(src, map(s -> findfirst(isequal(s), _names(src)), perm))  # PIRACY!

##### Standard Library

using LinearAlgebra

_value(A::Adjoint) = adjoint(_value(parent(A)))
_value(A::Transpose) = transpose(_value(parent(A)))
_value(A::Diagonal) = Diagonal(_value(parent(A)))

_named(A::LinearAlgebra.AdjointAbsVec, n::Symbol=:_, n2=nothing) = n2 !== nothing ? named(A, n2) : adjoint(named(parent(A), n))
_named(A::LinearAlgebra.TransposeAbsVec, n::Symbol=:_, n2=nothing) = n2 !== nothing ? named(A, n2) : transpose(named(parent(A), n))
_named(A::LinearAlgebra.AdjointAbsMat, nx::Symbol=:_, ny=:_) = adjoint(named(parent(A), ny, nx))
_named(A::LinearAlgebra.TransposeAbsMat, nx::Symbol=:_, ny=:_) = transpose(named(parent(A), ny, nx))
_named(A::Diagonal, n::Symbol) = Diagonal(named(parent(A),n))

for T in (:Symmetric, :Hermitian, :UpperTriangular, :LowerTriangular)
    @eval _named(A::$T, names::Symbol...) = $T(_named(parent(A), names...))
    @eval _value(A::$T) = $T(_value(parent(A)))
end

_MTs = [:AbstractNDA, :(Adjoint{<:Any,<:AbstractNDA}), :(Transpose{<:Any,<:AbstractNDA})]
for T in _MTs, T2 in _MTs
    @eval LinearAlgebra.mul!(C::AbstractNDA, A::$T, B::$T2, α::Number=true, β::Number=false) = begin 
        mul!(C.data, _value(A), _value(B), α, β); C end
end

LinearAlgebra.lu!(A::NDA, pivot=Val(true); check = true) = lu!(A.data, pivot; check)

LinearAlgebra.checksquare(A::AbstractNDA) = LinearAlgebra.checksquare(A.data) # avoids ==

Base.isassigned(A::AbstractArray, i::NamedInt...) = isassigned(A, map(_value, i)...)

using Statistics

Statistics._mean(f, A::AbstractArray, dims::Symbol) = Statistics._mean(f, A, _dim(A, dims))

"""
    named(A, :x, :y)

This is the friendly function for adding names. 
Unlike `NDA`, it will work through some wrappers, such as `Adjoint`. 
"""
named(A::AbstractArray, names...) = length(names) == ndims(A) ? NDA(A, map(Symbol, names)) : throw("wrong number of names!")  # should this un-wrap, _value(A)? then it won't check names?
named(A::AbstractArray, names::Tuple{Vararg{Symbol}}) = named(A, names...) # accept a tuple because you will mess up

for T in (:Adjoint, :Transpose, :Diagonal, :Symmetric, :Hermitian, :UpperTriangular, :LowerTriangular)
    @eval named(A::$T, names...) = _named(A, map(Symbol, names)...)
end

##### Printing

Base.showarg(io::IO, A::NDA, toplevel) = begin print(io, "named("); Base.showarg(io, _value(A), false); print(io, ", "); join(io, QuoteNode.(_names(A)), ", "); print(io, ")") end

# Base.dims2string(dt::Tuple{NamedInt}) = Base.dims2string((_value(dt[1]),))
# Base.dims2string(dt::Tuple{NamedInt{L}}) where {L} = string(L, "~", _value(dt[1]), "-element")  # a=2 -element named(::Vector...
Base.dims2string(dt::Tuple{NamedInt{L}}) where {L} = string("(", L, "⩽", _value(dt[1]), ")-element")  # a=2 -element named(::Vector...

Base.dims2string(xs::Tuple{NamedInt,Integer,Vararg{Integer}}) = _dims2string(xs)
Base.dims2string(xs::Tuple{Integer,NamedInt,Vararg{Integer}}) = _dims2string(xs)
Base.dims2string(xs::Tuple{NamedInt,NamedInt,Vararg{Integer}}) = _dims2string(xs)
# _dims2string(xs) = join([string(_name(x), "~", _value(x)) for x in xs], " × ")
_dims2string(xs) = join([string("(",_name(x), "⩽", _value(x),")") for x in xs], "×")

# NamedU{T} = Union{NDA{<:Any,T}, 
#     SubArray{T,<:Any,<:NDA}, PermutedDimsArray{T,<:Any,<:Any,<:Any,<:NDA},
#     Adjoint{T,<:NDA}, Transpose{T,<:NDA}} # this can't contain L as its order isn't fixed

# Base.show(io::IO, A::NamedU) = get(io, :typeinfo, Any) <: NDA ? show(io, parent(A)) : begin
#     print(io, "named("); show(io, _value(A)); print(io, ", ");
#     join(io, QuoteNode.(_names(A)), ", "); print(io, ")"); end

# function Base.print_matrix(io::IO, A::NamedU)
#     s1 = string("↓ :", _names(A)[1]) * "  "
#     ndims(A) == 2 && print(io, " "^Base.Unicode.textwidth(s1), "→ :", _names(A)[2], "\n")
#     ioc = IOContext(io, :displaysize => displaysize(io) .- (1, 0))
#     Base.print_matrix(ioc, _value(A), s1)
# end

function Base._show_nd_label(io::IO, A::AbstractArray, idxs::Tuple) # PIRACY! trivially more specialised
    print(io, "[:, :, ")
    for i in 1:length(idxs)
        n = _names(A)[i+2]
        print(io, n == :_ ? "" : "$n=", idxs[i])
        i == length(idxs) ? println(io, "] =") : print(io, ", ")
    end
end

# """
#     nameless(A)
#     # nameless(A, :x, :y)

# This is the friendly function for removing names.
# Or trying, for sufficiently many wrappers it may just do nothing.

# # If given (some) names, it will ensure the result matches.
# """
# nameless(A::AbstractArray) = _value(A)
