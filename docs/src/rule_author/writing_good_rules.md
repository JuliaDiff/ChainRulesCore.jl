# On writing good `rrule` / `frule` methods

## Code Style

Use named local functions for the `pullback` in an `rrule`.

```julia
# good:
function rrule(::typeof(foo), x)
    Y = foo(x)
    function foo_pullback(Ȳ)
        return NoTangent(), bar(Ȳ)
    end
    return Y, foo_pullback
end
#== output
julia> rrule(foo, 2)
(4, var"#foo_pullback#11"())
==#

# bad:
function rrule(::typeof(foo), x)
    return foo(x), x̄ -> (NoTangent(), bar(x̄))
end
#== output:
julia> rrule(foo, 2)
(4, var"##9#10"())
==#
```

While this is more verbose, it ensures that if an error is thrown during the `pullback` the [`gensym`](https://docs.julialang.org/en/v1/base/base/#Base.gensym) name of the local function will include the name you gave it.
This makes it a lot simpler to debug from the stacktrace.

## Use `ZeroTangent()` as the return value

The `ZeroTangent()` object exists as an alternative to directly returning `0` or `zeros(n)`.
It allows more optimal computation when chaining pullbacks/pushforwards, to avoid work.
They should be used where possible.

However, sometimes for performance reasons this is not ideal.
Especially, if it is to replace a scalar, and is in a type-unstable way.
It causes problems if mapping over such pullbacks/pushforwards.
This would be solved once [JuliaLang/julia#38241](https://github.com/JuliaLang/julia/issues/38241) has been addressed.

## Use `Thunk`s appropriately

If work is only required for one of the returned tangents, then it should be wrapped in a `@thunk` (potentially using a `begin`-`end` block).

If there are multiple return values, their computation should almost always be wrapped in a `@thunk`.

Do _not_ wrap _variables_ in a `@thunk`; wrap the _computations_ that fill those variables in `@thunk`:

```julia
# good:
∂A = @thunk(foo(x))
return ∂A

# bad:
∂A = foo(x)
return @thunk(∂A)
```
In the bad example `foo(x)` gets computed eagerly, and all that the thunk is doing is wrapping the already calculated result in a function that returns it.

Do not use `@thunk` if this would be equal or more work than actually evaluating the expression itself.
Examples being:
- The expression being a constant
- The expression is merely wrapping something in a `struct`, such as `Adjoint(x)` or `Diagonal(x)`
- The expression being itself a `thunk`
- The expression being from another `rrule` or `frule`;
  it would be `@thunk`ed if required by the defining rule already.
- There is only one derivative being returned, so from the fact that the user called
  `frule`/`rrule` they clearly will want to use that one.

## [Structs: constructors and functors](@id structs)

To define an `frule` or `rrule` for a _function_ `foo` we dispatch on the type of `foo`, which is `typeof(foo)`.
For example, the `rrule` signature would be like:

```julia
function rrule(::typeof(foo), args...; kwargs...)
    ...
    return y, foo_pullback
end
```

For a struct `Bar`,
```julia
struct Bar
    a::Float64
end

(bar::Bar)(x, y) = return bar.a + x + y # functor (i.e. callable object, overloading the call action)
```
we can define an `frule`/`rrule` for the `Bar` constructor(s), as well as any `Bar` [functors](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects).

### Constructors

To define an `rrule` for a constructor for a  _type_ `Bar` we need to be careful to dispatch only on `Type{Bar}`.
For example, the `rrule` signature for a `Bar` constructor would be like:
```julia
function ChainRulesCore.rrule(::Type{Bar}, a)
    Bar_pullback(Δbar) = NoTangent(), Δbar.a
    return Bar(a), Bar_pullback
end
```

Use `Type{<:Bar}` (with the `<:`) for non-concrete types, such that the `rrule` is defined for all subtypes.
In particular, be careful not to use `typeof(Bar)` here.
Because `typeof(Bar)` is `DataType`, using this to define an `rrule`/`frule` will define an `rrule`/`frule` for all constructors.

You can check which to use with `Core.Typeof`:

```julia
julia> function foo end
foo (generic function with 0 methods)

julia> typeof(foo)
typeof(foo)

julia> Core.Typeof(foob)
typeof(foo)

julia> typeof(Bar)
DataType

julia> Core.Typeof(Bar)
Type{Bar}

julia> abstract type AbstractT end

julia> typeof(AbstractT)
DataType

julia> Core.Typeof(AbstractT)
Type{AbstractT}
```

### Functors (callable objects)

In contrast to defining a rule for a constructor, it is possible to define rules for calling an instance of an object.
In that case, use `bar::Bar`, i.e.

```julia
function ChainRulesCore.rrule(bar::Bar, x, y)
    # Notice the first return is not `NoTangent()`
    Bar_pullback(Δy) = Tangent{Bar}(;a=Δy), Δy, Δy
    return bar(x, y), Bar_pullback
end
```
to define the rules.

## Ensure your pullback can accept the right types
As a rule the number of types you need to accept in a pullback is theoretically unlimitted, but practically highly constrained to be in line with the primal return type.
The three kinds of inputs you will practically need to accept one or more of: _natural tangents_, _structural tangents_, and _thunks_.
You do not in general have to handle `AbstractZero`s as the AD system will not call the pullback if the input is a zero, since the output will also be.
Some more background information on these types can be found in [the design notes](@ref manytypes).
In many cases all these tangents can be treated the same: tangent types overload a bunch of linear-operators, and the majority of functions used inside a pullback are linear operators.
If you find linear operators from Base/stdlibs that are not supported, consider opening an issue or a PR on the [ChainRulesCore.jl repo](https://github.com/JuliaDiff/ChainRulesCore.jl/).

### Natural tangents
Natural tangent types are the types you might feel the tangent should be, to represent a small change in the primal value.
For example, if the primal is a `Float32`, the natural tangent is also a `Float32`.
Slightly more complex, for a `ComplexF64` the natural tangent is again also a `ComplexF64`, we almost never want to use the structural tangent `Tangent{ComplexF64}(re=..., im=...)` which is defined.
For other cases, this gets a little more complicated, see below.
These are a purely human notion, they are the types the user wants to use because they make the math easy.
There is currently no formal definition of what constitutes a natural tangent, but there are a few heuristics.
For example, if a primal type `P` overloads subtraction (`-(::P,::P)`) then that generally returns a natural tangent type for `P`; but this is not required to be defined and sometimes it is defined poorly.

Common cases for types that represent a [vector-space](https://en.wikipedia.org/wiki/Vector_space) (e.g. `Float64`, `Array{Float64}`) is that the natural tangent type is the same as the primal type.
However, this is not always the case.
For example for a [`PDiagMat`](https://github.com/JuliaStats/PDMats.jl) a natural tangent is `Diagonal` since there is no requirement that a positive definite diagonal matrix has a positive definite tangent.
Another example is for a `DateTime`, any `Period` subtype, such as `Millisecond` or `Nanosecond` is a natural tangent.
There are often many different natural tangent types for a given primal type.
However, they are generally closely related and duck-type the same.
For example, for most `AbstractArray` subtypes, most other `AbstractArray`s (of right size and element type) can be considered as natural tangent types.

Not all types have natural tangent types.
For example there is no natural tangent for a `Tuple`.
It is not a `Tuple` since that doesn't have any method for `+`.
Similar is true for many `struct`s.
For those cases there is only a structural tangent.

### Structural tangents

Structural tangents are tangent types that shadow the structure of the primal type.
They are represented by the [`Tangent`](@ref) type.
They can represent any composite type, such as a tuple, or a structure (or a `NamedTuple`) etc.


!!! info "Do I have to support the structural tangents as well?"
    Technically, you might not actually have to write rules to accept structural tangents; if the AD system never has to decompose down to the level of `getfield`.
    This is common for types that don't support user `getfield`/`getproperty` access, and that have a lot of rules for the ways they are accessed (such cases include some `AbstractArray` subtypes).
    You really should support it just in case; especially if the primal type in question is not restricted to a well-tested concrete type.
    But if it is causing struggles, then you can leave it off til someone complains.

### Thunks

A thunk (either a [`Thunk`](@ref), or a [`InplaceableThunk`](@ref)), represents a delayed computation.
They can be thought of as a wrapper of the value the computation returns.
In this sense they wrap either a natural or structural tangent.

!!! warning "You should support AbstractThunk inputs even if you don't use thunks"
     Unfortunately the AD sytems do not know which rules support thunks and which do not.
     So all rules have to; at least if they want to play nicely with arbitrary AD systems.
     Luckily it is not hard: much of the time they will duck-type as the object they wrap.
     If not, then just add a [`unthunk`](@ref) after the start of your pullback.
     (Even when they do duck-type, if they are used multiple times then unthunking at the start will prevent them from being recomputed.)
     If you are using [`@thunk`](@ref) and the input is only needed for one of them then the `unthunk` should be in that one.
     If not, and you have a bunch of pullbacks you might like to write a little helper `unthunking(f) = x̄ -> f(unthunk(x̄))` that you can wrap your pullback function in before returning it from the `rrule`.
     Yes, this is a bit of boiler-plate, and it is unfortunate.
     Sadly, it is needed because if the AD wants to benefit it can't get that benifit unless things are not unthunked unnecessarily.
     Which eventually allows them in some cases to never be unthunked at all.
     There are two ways common things are never unthunked.
     One is if the unthunking happens inside a `@thunk` which is never unthunked itself because it is the tangent for a primal input that never has it's tangent queried.
     The second is if they are not unthunked because the rule does not need to know what is inside: consider the pullback for `identity`: `x̄ -> (NoTangent(), x̄)`.

## Use `@not_implemented` appropriately

You can use [`@not_implemented`](@ref) to mark missing tangents.
This is helpful if the function has multiple inputs or outputs, and you have worked out analytically and implemented some but not all tangents.

It is recommended to include a link to a GitHub issue about the missing tangent in the debugging information:
```julia
@not_implemented(
    """
    derivatives of Bessel functions with respect to the order are not implemented:
    https://github.com/JuliaMath/SpecialFunctions.jl/issues/160
    """
)
```

Do not use `@not_implemented` if the tangent does not exist mathematically (use `NoTangent()` instead).

Note: [ChainRulesTestUtils.jl](https://github.com/JuliaDiff/ChainRulesTestUtils.jl) marks `@not_implemented` tangents as "test broken".

## Use rule definition tools

Rule definition tools can help you write more `frule`s and the `rrule`s with less lines of code.
See [using rule definition tools](@ref ruletools) section for more details.

## Be careful about pullback closures calling other methods of themselves

Due to [JuliaLang/Julia#40990](https://github.com/JuliaLang/julia/issues/40990), a closure calling another (or the same) method of itself often comes out uninferable (and thus effectively type-unstable).
This can be avoided by moving the pullback definition outside the function, so that it is no longer a closure.
For example:

```julia
double_it(x::AbstractArray) = 2 .* x

function ChainRulesCore.rrule(::typeof(double_it), x)
    double_it_pullback(ȳ::AbstractArray) = (NoTangent(), 2 .* ȳ)
    double_it_pullback(ȳ::AbstractThunk) = double_it_pullback(unthunk(ȳ))
    return double_it(x), double_it_pullback
end
```
Ends up infering a return type of `Any`
```julia
julia> _, pullback = rrule(double_it, [2.0, 3.0])
([4.0, 6.0], var"#double_it_pullback#8"(Core.Box(var"#double_it_pullback#8"(#= circular reference @-2 =#))))

julia> @code_warntype pullback(@thunk([10.0, 10.0]))
Variables
  #self#::var"#double_it_pullback#8"
  ȳ::Core.Const(Thunk(var"#9#10"()))
  double_it_pullback::Union{}

Body::Any
1 ─ %1 = Core.getfield(#self#, :double_it_pullback)::Core.Box
│   %2 = Core.isdefined(%1, :contents)::Bool
└──      goto #3 if not %2
2 ─      goto #4
3 ─      Core.NewvarNode(:(double_it_pullback))
└──      double_it_pullback
4 ┄ %7 = Core.getfield(%1, :contents)::Any
│   %8 = Main.unthunk(ȳ)::Vector{Float64}
│   %9 = (%7)(%8)::Any
└──      return %9
```

This can be solved by moving the pullbacks outside the function so they are not closures, and thus to not run into this upstream issue.
In this case that is fairly simple, since this example doesn't close over anything (if it did then would need a closure calling an outside function that calls itself. See [this example](https://github.com/JuliaDiff/ChainRules.jl/blob/773039a2dc0a1938f61cf26012b1223c942bc18f/src/rulesets/LinearAlgebra/structured.jl#L107-L116).).

```julia
_double_it_pullback(ȳ::AbstractArray) = (NoTangent(), 2 .* ȳ)
_double_it_pullback(ȳ::AbstractThunk) = _double_it_pullback(unthunk(ȳ))

function ChainRulesCore.rrule(::typeof(double_it), x)
    return double_it(x), _double_it_pullback
end
```
This infers just fine:
```julia
julia> _, pullback = rrule(double_it, [2.0, 3.0])
([4.0, 6.0], _double_it_pullback)

julia> @code_warntype pullback(@thunk([10.0, 10.0]))
Variables
  #self#::Core.Const(_double_it_pullback)
  ȳ::Core.Const(Thunk(var"#7#8"()))

Body::Tuple{NoTangent, Vector{Float64}}
1 ─ %1 = Main.unthunk(ȳ)::Vector{Float64}
│   %2 = Main._double_it_pullback(%1)::Core.PartialStruct(Tuple{NoTangent, Vector{Float64}}, Any[Core.Const(NoTangent()), Vector{Float64}])
└──      return %2
```

Though in this particular case, it can also be solved by taking advantage of duck-typing and just writing one method.
Thus avoiding the call that confuses the compiler.
`Thunk`s duck-type as the type they wrap in most cases: including broadcast multiplication.

```julia
function ChainRulesCore.rrule(::typeof(double_it), x)
    double_it_pullback(ȳ) = (NoTangent(), 2 .* ȳ)
    return double_it(x), double_it_pullback
end
```
This infers perfectly.

## CAS systems are your friends.

It is very easy to check gradients or derivatives with a computer algebra system (CAS) like [WolframAlpha](https://www.wolframalpha.com/input/?i=gradient+atan2%28x%2Cy%29).
