# Pedagogical Example

This pedagogical example will show you how to write an `rrule`.
See [On writing good `rrule` / `frule` methods](@ref) section for more tips and gotchas.
If you want to learn about `frule`s, you should still read and understand this example as many concepts are shared, and then look for real world `frule` examples in ChainRules.jl.

## The primal

We define a struct `Foo`
```julia
struct Foo{T}
    A::Matrix{T}
    c::Float64
end
```
and a function that multiplies `Foo` with an `AbstractArray`:
```julia
function foo_mul(foo::Foo, b::AbstractArray)
    return foo.A * b
end
```
Note that field `c` is ignored in the calculation.

## The `rrule`

The `rrule` method for our primal computation should extend the `ChainRulesCore.rrule` function.
```julia
function ChainRulesCore.rrule(::typeof(foo_mul), foo::Foo{T}, b::AbstractArray) where T
    y = foo_mul(foo, b)
    function foo_mul_pullback(ȳ)
        f̄ = NoTangent()
        f̄oo = Tangent{Foo{T}}(; A=ȳ * b', c=ZeroTangent())
        b̄ = @thunk(foo.A' * ȳ)
        return f̄, f̄oo, b̄
    end
    return y, foo_mul_pullback
end
```

We can check this rule against a finite-differences approach using [`ChainRulesTestUtils`](https://github.com/JuliaDiff/ChainRulesTestUtils.jl):
```julia
julia> using ChainRulesTestUtils
julia> test_rrule(foo_mul, Foo(rand(3, 3), 3.0), rand(3, 3))
Test Summary:                                       | Pass  Total
test_rrule: foo_mul on Foo{Float64},Matrix{Float64} |   10     10
Test.DefaultTestSet("test_rrule: foo_mul on Foo{Float64},Matrix{Float64}", Any[], 10, false, false)
```

Now let's examine the rule in more detail:
```julia
function ChainRulesCore.rrule(::typeof(foo_mul), foo::Foo, b::AbstractArray)
    ...
    return y, foo_mul_pullback
end
```
The `rrule` dispatches on the `typeof` of the function we are writing the `rrule` for, as well as the types of its arguments.
Read more about writing rules for constructors and callable objects [here](@ref structs).
The `rrule` returns the primal result `y`, and the pullback function.
It is a _very_ good idea to name your pullback function, so that they are helpful when appearing in the stacktrace.
```julia
y = foo_mul(foo, b)
```
Computes the primal result.
It is possible to change the primal computation so that work can be shared between the primal and the pullback.
See e.g. [the rule for `sort`](https://github.com/JuliaDiff/ChainRules.jl/blob/a75193768775975fac5578c89d1e5f50d7f358c2/src/rulesets/Base/sort.jl#L19-L35), where the sorting is done only once.
```julia
function foo_mul_pullback(ȳ)
    ...
    return f̄, f̄oo, b̄
end
```
The pullback function takes in the tangent of the primal output (`ȳ`) and returns the tangents of the primal inputs.
Note that it returns a tangent for the primal function in addition to the tangents of primal arguments.

Finally, computing the tangents of primal inputs:
```julia
f̄ = NoTangent()
```
The function `foo_mul` has no fields (i.e. it is not a closure) and can not be perturbed.
Therefore its tangent (`f̄`) is a `NoTangent`.
```julia
f̄oo = Tangent{Foo}(; A=ȳ * b', c=ZeroTangent())
```
The struct `foo::Foo` gets a `Tangent{Foo}` structural tangent, which stores the tangents of fields of `foo`.

The tangent of the field `A` is `ȳ * b'`,

The tangent of the field `c` is `ZeroTangent()`, because `c` can be perturbed but has no effect on the primal output.
```julia
b̄ = @thunk(foo.A' * ȳ)
```
The tangent of `b` is `foo.A' * ȳ`, but we have wrapped it into a `Thunk`, a tangent type that represents delayed computation.
The idea is that in case the tangent is not used anywhere, the computation never happens.
Use [`InplaceableThunk`](@ref) if you are interested in [accumulating gradients in-place](@ref grad_acc).
Note that in practice one would also `@thunk` the `f̄oo.A` tangent, but it was omitted in this example for clarity.

As a final note, since `b` is an `AbstractArray`, its tangent `b̄` should be projected to the right subspace.
See the [`ProjectTo` the primal subspace](@ref projectto) section for more information and an example that motivates the projection operation.
