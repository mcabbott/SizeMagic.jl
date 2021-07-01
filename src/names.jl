export NamedInt, NamedArray, _value, _name, _names, _dim # for now!

abstract type AbstractNI <: Integer end

"""
    NamedInt(3, :μ) :: AbstractNI

An `Int` with a name attached! Using these for `size` is the
key idea of this package. 

Most operations discard the name. Only `+` and `==` preserve it,
and give an error if two names (neither `_`) don't match.
"""
struct NamedInt <: AbstractNI
    val::Int
    name::Symbol
    NamedInt(val::Integer, name::Symbol=:_) = new(Int(val), name)
end
_name(x::NamedInt) = x.name  # _name()::Symbol and _names()::Tuple are getters
_name(x::Integer) = :_
_value(x::NamedInt) = x.val
_value(x) = x

Base.promote_rule(::Type{<:AbstractNI}, ::Type{T}) where {T<:Number} = T
Base.promote_rule(::Type{<:AbstractNI}, ::Type{<:AbstractNI}) = Int
# Base.promote_rule(::Type{<:NamedInt{L}}, ::Type{<:NamedInt{L}}) where {L} = NamedInt{L}

Base.convert(::Type{T}, x::AbstractNI) where {T<:Number} = convert(T, _value(x))
Base.convert(::Type{NamedInt}, x::NamedInt) = x
(::Type{T})(x::AbstractNI) where {T<:Number} = begin T<:AbstractNI && @info "convert should combine?" T x; T(_value(x)) end # _combine()?
NamedInt(x::AbstractNI, n::Symbol=:_) = NamedInt(_value(x), _combine(n, _name(x)))

for op in (:-, :*, :&, :|, :xor, :<, :<=)
    @eval Base.$op(a::AbstractNI, b::AbstractNI) = $op(_value(a), _value(b))
end
for f in (:abs, :abs2, :sign, :-)
    @eval Base.$f(x::AbstractNI) = $f(_value(x))
end

Base.:(==)(a::NamedInt, b::NamedInt) = (_value(a) == _value(b)) && _compatable(_name(a), _name(b))

Base.:+(a::NamedInt, b::NamedInt) = _named(_value(a) + _value(b), _combine(_name(a), _name(b), true))
Base.:+(a::NamedInt, b::Integer) = _named(_value(a) + b, _name(a))
Base.:+(a::Integer, b::NamedInt) = _named(a + _value(b), _name(b))

_compatable(x::Symbol, y::Symbol) = (x === :_) | (y === :_) | (x === y)

_combine(x::Symbol, y::Symbol, strict::Bool=true) = x === :_ ? y : y === :_ ? x : x === y ? x :
    strict ? throw(DimensionMismatch("can't combine dimension names :$x and :$y")) : :_
# Surely Base._cs etc could share code here?

_named(x::Integer, s::Symbol) = NamedInt(x, s)
_named(x::Bool, ::Symbol) = x
_named(x, _) = x

Base.sum(t::Tuple{AbstractNI, Vararg{Integer}}) = sum(_value, t) # needed for show, within print_matrix

"""
    NamedArray(array, names) <: AbstractArray

Wrapper which attaches names, a tuple of symbols, to an array. 
The goal is to work like `NamedDimsArray`, with far less code,
by adding methods to lower-level functions to pass `NamedInt`s along.

The names are part of the value not the type, making type-stability easier.
"""
struct NamedArray{T,N,P} <: AbstractArray{T,N}
    data::P
    names::NTuple{N,Symbol}
    NamedArray(data::P, names::NTuple{N,Symbol}=ntuple(d->:_, N)) where {P<:AbstractArray{T,N}} where {T,N} =
        new{T,N,P}(data, names)
end

"""
    NamedDense(a::DenseArray, names) <: DenseArray

This is another wrapper, which turns one `StridedArray` into another one, 
with names. Its supertype is the only distinction from `NamedArray`.
"""
struct NamedDense{T,N,P} <: DenseArray{T,N}
    data::P
    names::NTuple{N,Symbol}
    NamedDense(data::P, names::NTuple{N,Symbol}=ntuple(d->:_, N)) where {P<:DenseArray{T,N}} where {T,N} =
        new{T,N,P}(data, names)
end

_named(A::AbstractArray{T,N}, names::Vararg{Symbol,N}) where {T,N} = NamedArray(A, names)
_named(A::DenseArray{T,N}, names::Vararg{Symbol,N}) where {T,N} = NamedDense(A, names)

NamedStruct{T,N,P} = Union{NamedArray{T,N,P}, NamedDense{T,N,P}}  # all have .data, .names

_value(A::NamedStruct) = _value(A.data) # keep looking inside?
_names(A::NamedStruct) = A.names
_names(A::AbstractArray) = map(_name, size(A)) # this works on wrappers

Base.size(A::NamedStruct) = map(NamedInt, size(A.data), A.names)
Base.size(A::NamedStruct, d::Integer) = d <= ndims(A) ? NamedInt(size(A.data,d), A.names[d]) : NamedInt(1, :_)
Base.axes(A::NamedStruct) = map(_named, axes(A.data), A.names)
Base.axes(A::NamedStruct, d::Integer) = _named(axes(A.data, d), d <= ndims(A) ? A.names[d] : :_)

Base.getindex(A::NamedStruct{T,N}, I::Vararg{Int,N}) where {T,N} = getindex(A.data, I...)
Base.setindex!(A::NamedStruct{T,N}, val, I::Vararg{Int,N}) where {T,N} = setindex!(A.data, val, I...)

Base.parent(A::NamedStruct) = A.data

Base.strides(A::NamedStruct) = strides(A.data)
Base.stride(A::NamedStruct, d::Int) = stride(A.data, d)
Base.unsafe_convert(::Type{Ptr{T}}, A::NamedStruct) where {T} = Base.unsafe_convert(Ptr{T}, A.data)
Base.elsize(::Type{<:NamedStruct{T,N,P}}) where {T,N,P} = Base.elsize(P)

for fun in (:getindex, :view, :dotview, :setindex!)  # Keyword indexing, via PIRACY!
    extra = fun == :setindex! ? (:val,) : ()
    @eval Base.$fun(A::AbstractArray, $(extra...); kw...) = 
        Base.$fun(A, $(extra...), ntuple(d -> get(kw, _names(A)[d], :), ndims(A))...) # matches repeated names, and _, and ignores other keywords -- not great.
end

for eq in [:(==), :isequal, :isapprox]  # doesn't work for x' == y'
    @eval Base.$eq(A::NamedStruct, B::NamedStruct) = $eq(_value(A), _value(A)) && (_names(A) == _names(B))
end

"""
    named(A, :x, :y) == named(A, "x", "y")

This is the friendly function for adding names. It will choose 
`NamedArray` or `NamedDense` as appropriate, and it will try to
place names underneath `Adjoint` etc. wrappers.
For adjoint vectors and `Diagonal`, supplying one name 
attaches this to the vector inside.
"""
named(A::AbstractArray, names...) = _named(A, map(Symbol, names)...)

_named(A::AbstractArray{T,N}, names::Symbol...) where {T,N} = throw("expected $N names for a $(typeof(A)), got $names")
_named(A::AbstractArray, names::Tuple) = _named(A, names...)


Base.similar(A::AbstractArray, ::Type{T}, dims::Tuple{NamedInt, Vararg{Integer}}) where {T} =
    _named(similar(A, T, map(_value, dims)), map(_name, dims))
Base.similar(::Type{A}, dims::Tuple{NamedInt, Vararg{Integer}}) where {A<:AbstractArray} =
    _named(similar(A, map(_value, dims)), map(_name, dims))

# (::Type{A})(::UndefInitializer, sz) where {A<:NamedArray{L,T,N,P}} where {L,T,N,P} = NamedArray(P(undef, sz), L)

(Array{T})(::UndefInitializer, a::NamedInt, bs::Integer...) where {T} = 
    NamedDense(Array{T}(undef, _value(a), map(_value, bs)...), (_name(a), map(_name, bs)...)) # dodgy but makes @einsum work, also Diagonal(named(1:10, :x)) .* 100

function Base.reshape(A::AbstractArray, shp::Tuple{<:NamedInt, Vararg{Integer}})
    data = reshape(_value(A), map(_value, shp))
    _named(data, map(_name, shp))  # todo -- check against previous, where sizes match? 
end
function Base._dropdims(A::NamedStruct, dims::Base.Dims)
    ax = Base._foldoneto((ds, d) -> d in dims ? ds : (ds..., axes(A,d)), (), Val(ndims(A)))
    reshape(_value(A), ax)
end

###### Broadcasting

const _NamedOneTo = Base.OneTo{NamedInt}
_name(r::_NamedOneTo) = _name(r.stop)
_value(r::_NamedOneTo) = Base.OneTo(_value(r.stop))

_named(r::Base.OneTo, s::Symbol) = Base.OneTo(_named(r.stop, s))

# Base.last(r::_NamedOneTo) = _value(r.stop) # for lastindex, means 2:end loses the name
# Base.UnitRange(r::_NamedOneTo) = UnitRange(_value(r)) # needed for show with last

Broadcast.axistype(a::_NamedOneTo, b::Base.OneTo) = a
Broadcast.axistype(a::Base.OneTo, b::_NamedOneTo) = Base.OneTo(_named(a.stop, _name(b)))
Broadcast.axistype(a::_NamedOneTo, b::_NamedOneTo) = Base.OneTo(_named(a.stop, _combine(_name(a),_name(b),true)))

# Base.Slice(a::_NamedOneTo) = Base.Slice(_value(a))

###### stuff?

# Base._cs(d, a::AbstractNI, b::AbstractNI) = _named(Base._cs(d, _value(a), _value(b)), _combine(_name(a), _name(b), true)) # used by cat(A, A'; dims=3), ==

# Base.promote_shape(ax::Tuple{_NamedOneTo,Vararg{<:Base.OneTo}}, bx::Base.Indices) = _promoteshape(ax, bx)
# _promoteshape(ax, bx) = 
# This is to make map(+, A, A') give an error, if == is not strict. May be desirably anyway, to promote? 

Base.to_shape(i::NamedInt) = i  # needed for hcat etc.
Base.to_shape(r::_NamedOneTo) = last(r)
_toshape(r::_NamedOneTo) = last(r)
_toshape(r::Base.OneTo) = NamedInt(last(r), :_)
Base.to_shape(ax::Tuple{_NamedOneTo,Base.OneTo,Vararg{<:Base.OneTo}}) = map(_toshape, ax)
Base.to_shape(ax::Tuple{Base.OneTo,_NamedOneTo,Vararg{<:Base.OneTo}}) = map(_toshape, ax)
Base.to_shape(ax::Tuple{_NamedOneTo,_NamedOneTo,Vararg{<:Base.OneTo}}) = map(_toshape, ax)

Base.unsafe_length(r::_NamedOneTo) = NamedInt(r.stop - zero(r.stop), _name(r.stop)) # for view
Base.unsafe_length(r::AbstractUnitRange{NamedInt}) where {L} = NamedInt(last(r) - first(r) + step(r), _name(last(r))) # a:b

_value(r::UnitRange{<:NamedInt}) = UnitRange(_value(r.start), _value(r.stop))
_value(v::SubArray) = SubArray(_value(v.parent), v.indices)
_value(A::PermutedDimsArray{T,N,P}) where {T,N,P} = PermutedDimsArray(_value(parent(A)), P)

_named(r::UnitRange, n::Symbol) = UnitRange(NamedInt(first(r),n), NamedInt(last(r),n))
(::Colon)(start::Integer, stop::NamedInt) = UnitRange(NamedInt(start, _name(stop)), stop) # wants Base.unsafe_length

Base.to_indices(A, inds::Tuple{_NamedOneTo, Vararg{Any}}, I::Tuple{AbstractVector, Vararg{Any}}) =
    (_named(I[1], _name(inds[1])), Base.to_indices(A, Base.tail(inds), Base.tail(I))...) # preserves names in view(A,1:2,:) -- but should we?

# Base._array_for(::Type{T}, itr::Base.Generator{_NamedOneTo}, ::Base.HasLength) where {T} = @show "yes"
# Base._array_for(::Type{T}, itr::Base.Generator{_NamedOneTo}, ::Base.HasShape{N}) where {T,N} = @show "yes"

##### Reductions & dims

Base._dropdims(A::AbstractArray, name::Symbol) = Base._dropdims(A, (_dim(A, name),)) # This is why sum() should keep names. Also, PIRACY!
Base._dropdims(A::AbstractArray, names::Tuple{Vararg{Symbol}}) = Base._dropdims(A, _dim(A, names)) # PIRACY!

# Base.reduced_indices(ax::Tuple{AbstractUnitRange{<:NamedInt}, Vararg{AbstractUnitRange}}, dim::Int) = ntuple(d -> d==dim ? Base.OneTo(1) : ax[d], length(ax)) # this seems too late, gives a stack overflow?

Base.reducedim_initarray(A::NamedStruct, region, init, ::Type{R}) where {R} =
    _named(Base.reducedim_initarray(A.data, _dim(A,region), init, R), _names(A))  # this makes sum(A, dims=1) work

Base.reducedim_initarray(A::AbstractArray, region::Union{Symbol, Tuple{Vararg{Symbol}}}, init, ::Type{R}) where {R} =
    Base.reducedim_initarray(A, _dim(A, region), init, R) # PIRACY, makes sum(A', dims=:x) work
Base.reducedim_initarray(A::NamedStruct, region::Union{Symbol, Tuple{Vararg{Symbol}}}, init, ::Type{R}) where {R} =
    Base.reducedim_initarray(A, _dim(A, region), init, R) # solve ambiguity

_dim(A::AbstractArray, d) = d
_dim(A::AbstractArray, s::Symbol) = begin d = findfirst(n -> n===s, _names(A)); d===nothing && throw("name $d not found in $(_names(A))"); d end
_dim(A::AbstractArray, ds::Tuple{Vararg{Symbol}}) = map(s -> _dim(A,s), ds)

Base.eachslice(A::NamedStruct; dims) = invoke(eachslice, Tuple{AbstractArray}, A; dims=_dim(A, dims))
Base.mapslices(f, A::NamedStruct; dims) = invoke(mapslices, Tuple{Any, AbstractArray}, f, A; dims=_dim(A, dims))
function Base.cumsum(A::NamedStruct{T}; dims) where {T}
    out = similar(A, Base.promote_op(Base.add_sum, T, T))
    cumsum!(out, A, dims=_dim(A,dims))
end
Base.cumprod(A::NamedStruct; dims) = accumulate(Base.mul_prod, A, dims=_dim(A, dims))

Base.PermutedDimsArrays.genperm(ax::Tuple{_NamedOneTo, Vararg{Any}}, perm::Tuple{Symbol, Vararg{Any}}) = begin
    axnames = map(_name, ax)
    map(s -> ax[findfirst(isequal(s), axnames)], perm)
end
Base.permutedims!(dest, src::AbstractArray, perm::Tuple{Vararg{Symbol}}) = permutedims!(dest, src, map(s -> findfirst(isequal(s), _names(src)), perm))  # PIRACY!
Base.PermutedDimsArray(src::AbstractArray, perm::Tuple{Vararg{Symbol}}) = PermutedDimsArray(src, map(s -> findfirst(isequal(s), _names(src)), perm))  # PIRACY!

##### Standard Library

using LinearAlgebra

# LinearAlgebra.lapack_size(t::AbstractChar, M::NamedStruct) = LinearAlgebra.lapack_size(t, M.data)
# LinearAlgebra.gemm_wrapper!(C::NamedStruct{T}, tA::AbstractChar, tB::AbstractChar,
#                        A::StridedVecOrMat{T}, B::StridedVecOrMat{T},
#                        _add = LinearAlgebra.MulAddMul()) where {T<:LinearAlgebra.BlasFloat} = begin
#     LinearAlgebra.gemm_wrapper!(C.data, tA, tB, _value(A), _value(B), _add); C end

LinearAlgebra.matmul2x2!(C::NamedStruct, tA, tB, A::AbstractMatrix, B::AbstractMatrix, _add::LinearAlgebra.MulAddMul) = 
    begin LinearAlgebra.matmul2x2!(C.data, tA, tB, _value(A), _value(B), _add); C end
LinearAlgebra.matmul3x3!(C::NamedStruct, tA, tB, A::AbstractMatrix, B::AbstractMatrix, _add::LinearAlgebra.MulAddMul) = 
    begin LinearAlgebra.matmul3x3!(C.data, tA, tB, _value(A), _value(B), _add); C end

_value(A::Adjoint) = adjoint(_value(parent(A)))
_value(A::Transpose) = transpose(_value(parent(A)))

_named(A::LinearAlgebra.AdjointAbsMat, nx::Symbol, ny::Symbol) = adjoint(_named(parent(A), ny, nx))
_named(A::LinearAlgebra.TransposeAbsMat, nx::Symbol, ny::Symbol) = transpose(_named(parent(A), ny, nx))

_named(A::LinearAlgebra.AdjointAbsVec, n::Symbol) = adjoint(_named(parent(A), n))
_named(A::LinearAlgebra.TransposeAbsVec, n::Symbol) = transpose(_named(parent(A), n))

_value(A::Diagonal) = Diagonal(_value(parent(A)))
_named(A::Diagonal, n::Symbol) = Diagonal(_named(A.diag, n)) 

for T in (:Symmetric, :Hermitian, :UpperTriangular, :LowerTriangular, :UnitUpperTriangular, :UnitLowerTriangular)
    extra = T in (:Symmetric, :Hermitian) ? [:(LinearAlgebra.sym_uplo(A.uplo))] : ()
    @eval _named(A::$T, nx::Symbol, ny::Symbol) = $T(_named(parent(A), nx, ny), $(extra...))
    @eval _value(A::$T) = $T(_value(parent(A)), $(extra...))
end

# _MTs = [:NamedStruct, :(Adjoint{<:Any,<:AbstractNamedArray}), :(Transpose{<:Any,<:AbstractNamedArray})]
# for T in _MTs, T2 in _MTs
#     @eval LinearAlgebra.mul!(C::AbstractNamedArray, A::$T, B::$T2, α::Number=true, β::Number=false) = begin 
#         mul!(C.data, _value(A), _value(B), α, β); C end
# end

# LinearAlgebra.lu!(A::NamedArray, pivot=Val(true); check = true) = lu!(A.data, pivot; check)

LinearAlgebra.checksquare(A::NamedStruct) = LinearAlgebra.checksquare(A.data) # avoids ==
LinearAlgebra.checksquare(A::LU{<:Any, <:NamedStruct}) = LinearAlgebra.checksquare(A.factors) # avoids ==

# SVD
# LinearAlgebra.copy_oftype(A::NamedStruct, ::Type{T}) where T = _named(LinearAlgebra.copy_oftype(A.data, T), _names(A)...) # not necc, but perhaps for eigen
LinearAlgebra.SVD(U::AbstractArray{T}, S::NamedStruct{Tr,1}, Vt::AbstractArray{T}) where {T,Tr} = SVD(U, S.data, Vt)

Base.isassigned(A::AbstractArray, i::NamedInt...) = isassigned(A, map(_value, i)...) # printing?

using Statistics

Statistics._mean(f, A::AbstractArray, dims::Symbol) = Statistics._mean(f, A, _dim(A, dims)) # PIRACY!

##### Printing

Base.showarg(io::IO, A::NamedStruct, toplevel) = begin print(io, "named("); Base.showarg(io, _value(A), false); print(io, ", "); join(io, QuoteNode.(_names(A)), ", "); print(io, ")") end

# Base.dims2string(dt::Tuple{NamedInt}) = Base.dims2string((_value(dt[1]),))
# Base.dims2string(dt::Tuple{NamedInt{L}}) where {L} = string(L, "~", _value(dt[1]), "-element")  # a=2 -element named(::Vector...
Base.dims2string(dt::Tuple{NamedInt}) = string("(", _name(dt[1]), "⩽", _value(dt[1]), ")-element")  # a=2 -element named(::Vector...

Base.dims2string(xs::Tuple{NamedInt,Integer,Vararg{Integer}}) = _dims2string(xs)
Base.dims2string(xs::Tuple{Integer,NamedInt,Vararg{Integer}}) = _dims2string(xs)
Base.dims2string(xs::Tuple{NamedInt,NamedInt,Vararg{Integer}}) = _dims2string(xs)
# _dims2string(xs) = join([string(_name(x), "~", _value(x)) for x in xs], " × ")
_dims2string(xs) = join([string("(",_name(x), "⩽", _value(x),")") for x in xs], "×")

# NamedU{T} = Union{NamedArray{<:Any,T}, 
#     SubArray{T,<:Any,<:NamedArray}, PermutedDimsArray{T,<:Any,<:Any,<:Any,<:NamedArray},
#     Adjoint{T,<:NamedArray}, Transpose{T,<:NamedArray}} # this can't contain L as its order isn't fixed

# Base.show(io::IO, A::NamedU) = get(io, :typeinfo, Any) <: NamedArray ? show(io, parent(A)) : begin
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


Base.show(io::IO, r::UnitRange{NamedInt}) = print(io, _value(first(r)), ":", _value(last(r))) # helps OffsetArrays printing!
# Base.show(io::IO, r::AbstractUnitRange{NamedInt}) = print(io, _value(first(r)), ":", _value(last(r)))

# """
#     nameless(A)
#     # nameless(A, :x, :y)

# This is the friendly function for removing names.
# Or trying, for sufficiently many wrappers it may just do nothing.

# # If given (some) names, it will ensure the result matches.
# """
# nameless(A::AbstractArray) = _value(A)
