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
(y⩽3)×(x⩽3) adjoint(named(::Matrix{Int64}, :x, :y)) with eltype Int64:
  71  -65  -10
  60   54  -84
 -90  -33  -81

julia> axes(A')
(Base.OneTo(NamedInt(3, :y)), Base.OneTo(NamedInt(3, :x)))

julia> A isa StridedMatrix
true

julia> A' * A
(y⩽3)×(y⩽3) named(::Matrix{Int64}, :y, :y):
  9366   1590  -3435
  1590  13572   -378
 -3435   -378  15750

julia> sum(cbrt, A, dims=:x)
(x⩽1)×(y⩽3) named(::Matrix{Float64}, :x, :y):
 -2.03434  3.31511  -12.0157

julia> hcat(A, A)
(x⩽3)×(y⩽6) named(::Matrix{Int64}, :x, :y):
  71   60  -90   71   60  -90
 -65   54  -33  -65   54  -33
 -10  -84  -81  -10  -84  -81

julia> hcat(A, A')
ERROR: ArgumentError: number of rows of each array must match (got (NamedInt(3, :x), NamedInt(3, :y)))
Stacktrace:
 [1] _typed_hcat(#unused#::Type{Int64}, A::Tuple{SizeMagic.NamedDense{Int64, 2, Matrix{Int64}}, Adjoint{Int64, SizeMagic.NamedDense{Int64, 2, Matrix{Int64}}}})
   @ Base ./abstractarray.jl:1573
```
