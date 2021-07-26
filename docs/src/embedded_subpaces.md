# Types that represent embedded subspaces
_Taking Types Representing Embedded Subspaces Seriously™_

To paraphrase Stefan Karpinski: _"This does mean treating sparse matrixes not just as a representation of dense matrixes, but the alternative is just too horrible."_


Consider the following possible `rrule` for `*`
```julia
function rrule(::typeof(*), a, b)
    mul_pullback(ȳ) = (NoTangent(), ȳ*b', a'*ȳ)
    return (a * b), mul_pullback
end
```

This seems perfectly reasonable for floats:
```julia
julia> _, pb = rrule(*, 2.0, 3.0)
(6.0, var"#mul_pullback#10"{Float64, Float64}(2.0, 3.0))

julia> pb(1.0)
(NoTangent(), 3.0, 2.0)
```
and for matrixes
```julia
julia> _, pb = rrule(*, [1.0 2.0; 3.0 4.0], [10.0 20.0; 30.0 40.0])
([70.0 100.0; 150.0 220.0], var"#mul_pullback#10"{Matrix{Float64}, Matrix{Float64}}([1.0 2.0; 3.0 4.0], [10.0 20.0; 30.0 40.0]))

julia> pb([1.0 0.0; 0.0 1.0])
(NoTangent(), [10.0 30.0; 20.0 40.0], [1.0 3.0; 2.0 4.0])
```

and even for complex numbers (assuming you like conjugation [which we do](@ref complexfunctions))
```julia
julia> _, pb = rrule(*, 0.0 + im, 1.0 + im)
(-1.0 + 1.0im, var"#mul_pullback#10"{ComplexF64, ComplexF64}(0.0 + 1.0im, 1.0 + 1.0im))

julia> pb(1.0)
(NoTangent(), 1.0 - 1.0im, 0.0 - 1.0im)

julia> pb(1.0im)
(NoTangent(), 1.0 + 1.0im, 1.0 + 0.0im)
```

So far everything is wonderful.
Isn't linear algebra great? We get this nice code that generalizes to all kinds of vector spaces.

What if we start mixing it up a bit.
Let's try a real amd a complex
```julia
julia> _, pb = rrule(*, 2.0, 3.0im)
(0.0 + 6.0im, var"#mul_pullback#10"{Float64, ComplexF64}(2.0, 0.0 + 3.0im))

julia> pb(1.0)
(NoTangent(), 0.0 - 3.0im, 2.0)

julia> pb(1.0im)
(NoTangent(), 3.0 + 0.0im, 0.0 + 2.0im)
```

That's _an_ answer.
It's consistent with treating the reals as being an embedded subspace of the complex numbers.
i.e. treating `2.0` as actually being `2.0 + 0im`.
It doesn't feel great that we have escaped from the real type in this way.
But it's not wrong as such.

Over to matrixes, lets try as `Diagonal` with a `Matrix`
```julia
julia> _, pb = rrule(*, Diagonal([1.0, 2.0]), [1.0 2.0; 3.0 4.0])
([1.0 2.0; 6.0 8.0], var"#mul_pullback#10"{Diagonal{Float64, Vector{Float64}}, Matrix{Float64}}([1.0 0.0; 0.0 2.0], [1.0 2.0; 3.0 4.0]))

julia> pb([1.0 0.0; 0.0 1.0])
(NoTangent(), [1.0 3.0; 2.0 4.0], [1.0 0.0; 0.0 2.0])
```

This is also _an_ answer.
This seems even worse though: no only has it escaped the `Diagonal` type, it has even escaped the subspace it represents.

Further, it is inconsistent with what we would get had we AD'd the function for `*(::Diagonal, ::Matrix)` directly.
[That primal function](https://github.com/JuliaLang/julia/blob/f7f46af8ff39a1b4c7000651c680058e9c0639f5/stdlib/LinearAlgebra/src/diagonal.jl#L224-L245) boils down to:
```julia
*(a::Diagonal, b::AbstractMatrix) = a.diag .* b
```
By reading that primal method, we know that that ADing that method would have zeros on the off-diagonals, because they are never even accessed
(a similar argument can be made for the complex part of a real number).

---

Consider the following possible `rrule` for `sum`
```julia
function rrule(::typeof(sum), x::AbstractArray)
    sum_pullback(ȳ) = (NoTangent(), fill(ȳ, size(x)))
    return sum(x), sum_pullback
end
```

This seems all well and good at first:
```julia
julia> _, pb = rrule(sum, [1.0 2.0; 3.0 4.0])
(10.0, var"#sum_pullback#9"{Matrix{Float64}}([1.0 2.0; 3.0 4.0]))

julia> pb(1.0)
(NoTangent(), [1.0 1.0; 1.0 1.0])
```

But now consider:
```julia
julia> _, pb = rrule(sum, Diagonal([1.0, 2.0]))
(3.0, var"#sum_pullback#9"{Diagonal{Float64, Vector{Float64}}}([1.0 0.0; 0.0 2.0]))

julia> pb(1.0)
(NoTangent(), [1.0 1.0; 1.0 1.0])
```
That's not right -- not if us saying this was `Diagonal` meant anything.
If you try and use that dense matrix, to do gradient descent on your `Diagonal` input, you will get a non-diagonal result:
`[2.0 1.0; 1.0 2.0]`.
You have escape the subspace that the diagonal type represents.