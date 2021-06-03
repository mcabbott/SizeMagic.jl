export Axis, AxisInt, LAA, _labels, _axis

"""
    Axis(1:10, 'a':'j', :alpha) :: AbstractUnitRange

This is the structure to store axis labels.
It will be returned directly `axes(::AxA)`.
"""
struct Axis{L,T,A} <: AbstractUnitRange{T}
    data::A
    Axis(r::A, n::Symbol=:_) where {A<:AbstractUnitRange{T}} where {T} = new{n,T,A}(r)
end

# _labels(x::AxisInt) = x.axis # no good fallback possible!
# _labels(x) = nothing


for f in (:first, :last, :length, :size)
    @eval Base.$f(r::Axis) = $f(r.data)
end


"""
    AxisInt(::Axis) :: AbstractNI

To help `Axis` propagate, `size(::AxA)` returns special integers 
with a fragile payload. Many overloads are for `AbstractNI` thus
shared with (simpler) `NamedInt`.
"""
struct AxisInt{L,A} <: AbstractNI{L}
    axis::A
    AxisInt(ax::A) where {A<:Axis{L}} where {L} = new{L,A}(ax)
end
AxisInt(r::AbstractUnitRange) = AxisInt(Axis(r))
AxisInt(i::Integer) = AxisInt(Axis(Base.OneTo(i)))

_value(x::AxisInt) = length(x.axis)

Base.show(io::IO, x::AxisInt{L}) where {L} = print(io, x.val, "ᵡ")
Base.show(io::IO, ::MIME"text/plain", x::NamedInt{L}) where {L} =
    print(io, "AxisInt{$L}(", _value(x), ", ...)")

const ᵡ = AxisInt(Axis(_=Base.OneTo(1)))

"""
    LAA(array, labels, names)

The goal is to work like `AxisArrays` & `AxisKeys` with less code
"""
struct LAA{L,T,N,P,A} <: AbstractNDA{L,T,N}
    data::P
    axes::A
    LAA(data::P, labels::Tuple{Vararg{AbstractVector,N}}, names::NTuple{N,Symbol}) where {P<:AbstractArray{T,N}} where {T,N} = 
        new{names,T,N,P}(data)
end
Base.size(A::LAA) = map(AxisInt, A.axes)
Base.axes(A::LAA) = A.axes
Base.parent(A::LAA) = A.data

# Base.getindex(A::LAA{L,T,N}, I::Vararg{Int,N}) where {L,T,N} = getindex(A.data, I...)
# Base.setindex!(A::LAA{L,T,N}, val, I::Vararg{Int,N}) where {L,T,N} = setindex!(A.data, val, I...)


"""
    labelled(array, labels...)
    labelled(array; name = label, ...)

This is meant to be the friendly fuction for adding axis labels, and perhaps names.
"""
labelled(A::AbstractArray, labels::AbstractVector...) = LAA(A, labels, _names(A))
labelled(A::AbstractArray; kw...) = LAA(A, keys(kw), values(kw))


