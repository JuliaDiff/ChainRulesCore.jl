# [Gradient Accumulation](@id grad_acc)

Consider some function
$$f(x) = g(x) + h(x)$$.
If we would like the derivative of $f$ with respect to $x$ we must compute it for each part and then sum them, i.e.
$$\frac{\partial f}{\partial x} = \frac{\partial g}{\partial x} + \frac{\partial h}{\partial x}$$.
In general, we must accumulate (sum) gradients from each sub-part of a program where a variable is used.


Consider for example:
```julia
function sum_first_and_second(X::Array{Float64})
    a = X[1]
    b = X[2]
    y = a + b
    return y
end
```
The AD software must transform that into something which repeatedly sums up the gradient of each part:
`X̄ = ā + b̄`.

This requires that all tangent types `D` must implement `+`: `+(::D, ::D)::D`.

We can note that in this particular case `ā` and `b̄` will both be arrays.
This operation (`X̄ = ā + b̄`) will allocate one array to hold `ā`, another one to hold `b̄`, and a third one to hold `ā + b̄`.
This is three allocations.
Allocations are not free, they increase the time the program takes to run by a nontrivial amount, even with a good allocator and a good garbage collector.

### Maybe-mutating accumulation (`add!!`)
We can note that in the above that neither `ā` nor `b̄` are ever used again after accumulating to get `X̄`.
Furthermore, `Array`s are mutable.
That means we could over-write either `ā` or `b̄` and use the result as `X̄`:

```julia
ā .+= b̄
X̄ = ā
```

This cuts our allocations down to 2, just `ā` and `b̄`.

However, we have a bit of a problem that not all types are mutable, so this pattern is hard to apply in general.
To deal with that ChainRulesCore provides [`add!!`](@ref).
Per the [BangBang.jl](https://github.com/JuliaFolds/BangBang.jl) convention, this is a maybe mutating addition.
It may mutate its first argument (if it is mutable), but it will definitely return the correct result.
We would write using that as `X̄ = add!!(ā, b̄)`: which would in this case give us just 2 allocations.
AD systems can generate `add!!` instead of `+` when accumulating gradient to take advantage of this.

### Inplaceable Thunks (`InplaceableThunks`) avoid allocating values in the first place.
We got down to two allocations from using [`add!!`](@ref), but can we do better?
We can think of having a tangent type which acts on a partially accumulated result, to mutate it to contain its current value plus the partial derivative being accumulated.
Rather than having an actual computed value, we can just have a thing that will act on a value to perform the addition.
Let's illustrate it with our example.

`b̄` is the partial for `X[2]` and its value can be computed by:

```julia
b̄ = zeros(size(X))
b̄[2] = ȳ  # the scalar sensitivity of the `sum_first_and_second` output
```
`b̄` is a matrix entirely of zeros, except for at the index `2`, where it is set to the output sensitivity `ȳ`.
`ā` is similar, except with the non-zero at index `1`.

What is the action of `b̄` upon `ā`, to get the same result as `X̄ = add!!(ā, b̄)` (or `X̄ = ā + b̄` for that matter)?
It is:

```julia
function b̄_add!(ā)
    ā[2] += ȳ
    return ā
end
```
We don't need to worry about all those zeros since `x + 0 == x`.

[`InplaceableThunk`](@ref) is the type we have to represent derivatives as gradient accumulating actions.
We must note that to do this we do need a value form of `ā` for `b̄` to act upon.
For this reason every inplaceable thunk has both a `val` field holding the value representation, and a `add!` field holding the action representation.
The `val` field use a plain [`Thunk`](@ref) to avoid the computation (and thus allocation) if it is unused.

!!! note "Do we need both representations?"
    Right now every [`InplaceableThunk`](@ref) has two fields that need to be specified.
    The value form (represented as a the [`Thunk`](@ref) typed field), and the action form (represented as the `add!` field).
    It is possible in a future version of ChainRulesCore.jl we will work out a clever way to find the zero tangent for arbitrary primal values.
    Given that, we could always just determine the value form from `inplaceable.add!(zero_tangent(primal))`.
    There are some technical difficulties in finding the zero tangents, but this may be solved at some point.


The `+` operation on `InplaceableThunk`s is overloaded to [`unthunk`](@ref) that `val` field to get the value form.
Where as the [`add!!`](@ref) operation is overloaded to call `add!` to invoke the action.

With `getindex` defined to return an `InplaceableThunk`, we now get to `X̄ = add!!(ā, b̄)` requires only a single allocation.
This allocation occurs when `unthunk`ing `ā`, which is then mutated to become `X̄`.
This is basically as good as we can get: if we want `X̄` to be an `Array` then at some point we need to allocate that array.

!!! note "Can we do more? Deferred accumulation"
    We could keep going further to drop allocations if we really wanted.
    If we didn't care about `X̄` being an `Array` then we could defer its computation too.
    `X̄ = @thunk add!!(ā, b̄)`.
    This kind of deferral will work fine and you can keep chaining it.
    It does start to burn stack space, and might make the compiler's optimization passes cry.
    But it's valid and should work fine.

### Examples of InplaceableThunks

#### `getindex`

The aforementioned `getindex` is really the poster child for this.
Consider something like:
```julia
function mysum(X::Array{Float64})
    total = 0.0
    for i in eachindex(X)
        total += X[i]
    end
    return total
end
```
If one only has value representation of derivatives one ends up having to allocate a derivative array for every single element of the original array `X`.
That's terrible.
On the other hand, with the action representation that `InplaceableThunk`s provide, there is just a single `Array` allocated.
One can see [the `getindex` rule in ChainRules.jl for the implementation](https://github.com/JuliaDiff/ChainRules.jl/blob/v0.7.49/src/rulesets/Base/indexing.jl).


#### matmul etc (`*`)
Multiplication of scalars/vectors/matrices of compatible dimensions can all also have their derivatives represented as an `InplaceableThunk`.
These tend to pivot around that `add!` action being defined along the lines of:
`X̄ -> mul!(X̄, A', Ȳ, true, true)`.
Where 5-arg `mul!` is the in place [multiply-add operation](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.mul!).
`mul!(X̄, A', Ȳ, true, true)` has the same effect as `(X̄ .+= A'*Ȳ)` but avoids allocating  the matrix  `A'*Ȳ`
This is one of the fundamental operations provided by BLAS -- including the application of the conjugate transpose.
e.g. the Matrix-Matrix form is [`GEMM` (GEneralized Matrix-Matrix Multiplication)](http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html#gaeda3cbd99c8fb834a60a6412878226e1),
the Matrix-Vector form is [`GEMV` (GEneralized Matrix-Vector Multiplication)](http://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gadd421a107a488d524859b4a64c1901a9.html#gadd421a107a488d524859b4a64c1901a9) etc.
Under the hood doing it out of place is going to call one of these methods anyway, but on a freshly allocated output array.
So we are going to hit a very efficient implementation and get the addition for free.


One can see [the `*` rules in ChainRules.jl for the implementations](https://github.com/JuliaDiff/ChainRules.jl/blob/v0.7.49/src/rulesets/Base/arraymath.jl#L22-L95)
