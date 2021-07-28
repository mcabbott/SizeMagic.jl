# SizeMagic.jl

This is an experiment to see how little code might be needed to reproduce what [NamedDims.jl](https://github.com/invenia/NamedDims.jl) does, and (although not yet) what AxisArrays / AxisKeys / AxisIndices / DimensionalData do.

The best-behaved citizens are those functions which make a new array via `similar(A, T, axes(A))`; these usually automatically propogate OffsetArrays.jl. But many functions go instead via `similar(A, T, size(A))`, for instance all of LinearAlgebra.jl. The basic ideas of this package are:

* If `size(A)` returns a special `NamedInt`, then this information can often propagate through Julia code with no explicit handling. This includes broadcasting, comprehensions, `cat`. It also works from within other wrappers such as `Adjoint`.

* If in addition to `struct NamedArray <: AbstractArray`  we define a wrapper `struct NamedDense <: DenseArray`, and forward funcitons like `pointer`, then things like BLAS can also work with no explicit handling. And no method ambiguities from `*`.

Examples:

```julia
julia> using SizeMagic

julia> A = named(rand(Int8,3,3), :x, :y) .+ 0;

julia> A'
3×3 adjoint(named(::Matrix{Int64}, :x, :y)) with eltype Int64:
 -26  117  -38
  69   71   87
  92  126   -6

julia> axes(A')
(Base.OneTo(NamedInt(3, :y)), Base.OneTo(NamedInt(3, :x)))

julia> A isa StridedMatrix
true

julia> A' * A
3×3 named(::Matrix{Int64}, :y, :y):
 15809   3207  12578
  3207  17371  14772
 12578  14772  24376

julia> sum(A, dims=:x)
1×3 named(::Matrix{Int64}, :x, :y):
 53  227  212
```