using Test, SizeMagic
using SizeMagic: NamedInt, NDA

macro hasnames(args...)
    names, ex = map(quotenode, Base.front(args)), last(args)
    res, tup = gensym(), Symbol(string("names(", ex, ")"))
    out = quote
        $res = $(esc(ex))
        $tup = $SizeMagic._names($res)
    end
    for (i,n) in enumerate(names)
        push!(out.args, :(@test $tup[$i] == $n))
    end
    out
end
quotenode(s::Symbol) = QuoteNode(s)
quotenode(q::QuoteNode) = q

@testset "NamedInt" begin
    a = NamedInt(2, :a)
    @test a+1 isa NamedInt{:a}
    @test a+a isa NamedInt{:a}
    @test a*a === 4
    @test a^2 === 4
    @test (a<2) === false

    b = NamedInt(3, :b)
    @test a < b
    @test a * b === 6
    @test_throws Any a + b
    # @test_throws Any a == b
end

@testset "array: propagation" begin
    A = named(Int.(rand(Int8,3,3)), :x, :y)

    # Broadcasting
    @hasnames x y  1 .+ A
    @hasnames y x  1 .+ A'
    @hasnames x y  A .+ 2 .* A
    @test_throws Any A .+ A'

    # Indexing
    @hasnames x  A[:,1]

    v = @view A[:,1]
    @hasnames x  1 .+ v
    @hasnames x x  v ./ v'
    @hasnames x  v .+ (1,2,3)

    @hasnames x y @view A[:,2:end] # keeps y
    @test_skip @hasnames x y @view A[:,1:2] # forgets y
    @hasnames x _  A[:,1,:]

    # Reductions
    @hasnames x y  sum(A, dims=1)
    @hasnames x dropdims(sum(A, dims=2), dims=2)

    # Functions which just work
    @hasnames x y  hcat(A,A)
    @hasnames x y _  cat(A,A; dims=(1,2,3)) 
    @hasnames y x permutedims(A)
    @test_skip @hasnames _ x permutedims(v) # to_shape stackoverflow
    @test_throws Any vcat(A, A')
    @test_throws Any cat(A, A', dims=3)
    @test_throws Any map(+, A, A')

    @hasnames x y  mapslices(identity, A, dims=1)
    @hasnames x _  reduce(hcat, [x .+ 1 for x in eachcol(A)])
    @hasnames y  map(sum, eachcol(A))

    # Generators
    @hasnames x y  [x^2 for x in A]
    @hasnames x  [i^2 for i in 1:size(A,1)]

    # Permutations
    P = PermutedDimsArray(A, (2,1))
    @hasnames y x  1 .+ P
    @hasnames x y  1 .+ P'
    @hasnames y x  sum(P, dims=2)
    @hasnames x dropdims(sum(P', dims=2), dims=2)
    @test_skip P != A                   # throws, but shouldn't!
end

@testset "array: named access" begin
    A = named(Int.(rand(Int8,3,3)), :x, :y)
    @test A[y=1, x=1] === A[1]
    @test A[x=2] == A[2,:]

    @test (A[x=2] .= (1:3)) == 1:3  # dotview
    @test A[2,:] == 1:3
    @test (A[x=3, y=3] = 99) == 99  # setindex!
    @test A[3,3] == 99

    @test sum(A, dims=:x) == sum(A, dims=1)
    @test sum(A, dims=(:x,:y)) == sum(A, dims=(1,2))
    # @test prod(A', dims=:x) == prod(A', dims=1)
    P = PermutedDimsArray(A, (2,1))
    # sum(P, dims=:y) == sum(P, dims=2)
end

#=


A[:,1,:] # StackOverflowError

dropdims(sum(A, dims=1), dims=1)

qr(M)

=#

using LinearAlgebra, Statistics

@testset "LinearAlgebra" begin
    x = named(rand(3), :x)
    y = named(rand(3), :y)
    M = named(rand(3,3), :x, :y)
    @hasnames x  M * y
    @hasnames x x M * M'
    @hasnames y  x' * M

    @test det(M) isa Number

    D = named(Diagonal(rand(3)), :x)
    @hasnames x y  D * M
    @hasnames x  D * x

    @test_throws Any x + y
    @test_throws Any x' * y
    @test_throws Any D * y
    @test_throws Any dot(x,y)
end

@testset "Statistics" begin
    A = named(Int.(rand(Int8,3,3)), :x, :y)
    mean(A, dims=:x)
    mean(A', dims=:x)
    # std(A, dims=:x)

end
