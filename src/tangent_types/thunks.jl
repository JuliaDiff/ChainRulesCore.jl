abstract type AbstractThunk <: AbstractTangent end

struct MutateThunkException <: Exception end

function Base.showerror(io::IO, e::MutateThunkException)
    print(io, "Tried to mutate a thunk, this is not supported. `unthunk` it first.")
    return nothing
end

Base.Broadcast.broadcastable(x::AbstractThunk) = broadcastable(unthunk(x))

@inline function Base.iterate(x::AbstractThunk)
    val = unthunk(x)
    el_and_state = iterate(val)
    el_and_state isa Nothing && return nothing
    element, state = el_and_state
    return element, (val, state)
end

@inline function Base.iterate(::AbstractThunk, (underlying_object, state))
    next = iterate(underlying_object, state)
    next === nothing && return nothing
    element, new_state = next
    return element, (underlying_object, new_state)
end

Base.first(x::AbstractThunk) = first(unthunk(x))
Base.last(x::AbstractThunk) = last(unthunk(x))
Base.tail(x::AbstractThunk) = Base.tail(unthunk(x))

Base.:(==)(a::AbstractThunk, b::AbstractThunk) = unthunk(a) == unthunk(b)

Base.:(-)(a::AbstractThunk) = -unthunk(a)
Base.:(-)(a::AbstractThunk, b) = unthunk(a) - b
Base.:(-)(a, b::AbstractThunk) = a - unthunk(b)
Base.:(-)(a::AbstractThunk, b::AbstractThunk) = unthunk(a) - unthunk(b)
Base.:(/)(a::AbstractThunk, b) = unthunk(a) / b
Base.:(/)(a, b::AbstractThunk) = a / unthunk(b)
Base.:(/)(a::AbstractThunk, b::AbstractThunk) = unthunk(a) / unthunk(b)

# Fix method ambiguity issue
Base.:/(a::AbstractZero, ::AbstractThunk) = a

Base.real(a::AbstractThunk) = real(unthunk(a))
Base.imag(a::AbstractThunk) = imag(unthunk(a))
Base.Complex(a::AbstractThunk) = Complex(unthunk(a))
Base.Complex(a::AbstractThunk, b::AbstractThunk) = Complex(unthunk(a), unthunk(b))

Base.mapreduce(f, op, a::AbstractThunk; kws...) = mapreduce(f, op, unthunk(a); kws...)
function Base.mapreduce(f, op, itr, a::AbstractThunk; kws...)
    return mapreduce(f, op, itr, unthunk(a); kws...)
end
Base.sum(a::AbstractThunk; kws...) = sum(unthunk(a); kws...)
Base.sum!(r, A::AbstractThunk; kws...) = sum!(r, unthunk(A); kws...)

Base.fill(a::AbstractThunk, b::Integer) = fill(unthunk(a), b)
Base.vec(a::AbstractThunk) = vec(unthunk(a))
Base.reshape(a::AbstractThunk, args...) = reshape(unthunk(a), args...)
Base.reverse(a::AbstractThunk, args...; kwargs...) = reverse(unthunk(a), args...; kwargs...)
Base.getindex(a::AbstractThunk, args...) = getindex(unthunk(a), args...)
Base.setindex!(a::AbstractThunk, value, key...) = throw(MutateThunkException())
Base.selectdim(a::AbstractThunk, args...) = selectdim(unthunk(a), args...)

LinearAlgebra.Array(a::AbstractThunk) = Array(unthunk(a))
LinearAlgebra.Matrix(a::AbstractThunk) = Matrix(unthunk(a))
LinearAlgebra.Diagonal(a::AbstractThunk) = Diagonal(unthunk(a))
LinearAlgebra.LowerTriangular(a::AbstractThunk) = LowerTriangular(unthunk(a))
LinearAlgebra.UpperTriangular(a::AbstractThunk) = UpperTriangular(unthunk(a))
LinearAlgebra.Symmetric(a::AbstractThunk, uplo=:U) = Symmetric(unthunk(a), uplo)
LinearAlgebra.Hermitian(a::AbstractThunk, uplo=:U) = Hermitian(unthunk(a), uplo)

function LinearAlgebra.diagm(
    kv::Pair{<:Integer,<:AbstractThunk}, kvs::Pair{<:Integer,<:AbstractThunk}...
)
    return diagm((k => unthunk(v) for (k, v) in (kv, kvs...))...)
end
function LinearAlgebra.diagm(
    m, n, kv::Pair{<:Integer,<:AbstractThunk}, kvs::Pair{<:Integer,<:AbstractThunk}...
)
    return diagm(m, n, (k => unthunk(v) for (k, v) in (kv, kvs...))...)
end

LinearAlgebra.tril(a::AbstractThunk) = tril(unthunk(a))
LinearAlgebra.tril(a::AbstractThunk, k) = tril(unthunk(a), k)
LinearAlgebra.triu(a::AbstractThunk) = triu(unthunk(a))
LinearAlgebra.triu(a::AbstractThunk, k) = triu(unthunk(a), k)
LinearAlgebra.tr(a::AbstractThunk) = tr(unthunk(a))
LinearAlgebra.cross(a::AbstractThunk, b) = cross(unthunk(a), b)
LinearAlgebra.cross(a, b::AbstractThunk) = cross(a, unthunk(b))
LinearAlgebra.cross(a::AbstractThunk, b::AbstractThunk) = cross(unthunk(a), unthunk(b))
LinearAlgebra.dot(a::AbstractThunk, b) = dot(unthunk(a), b)
LinearAlgebra.dot(a, b::AbstractThunk) = dot(a, unthunk(b))
LinearAlgebra.dot(a::AbstractThunk, b::AbstractThunk) = dot(unthunk(a), unthunk(b))

LinearAlgebra.ldiv!(a, b::AbstractThunk) = throw(MutateThunkException())
LinearAlgebra.rdiv!(a::AbstractThunk, b) = throw(MutateThunkException())

LinearAlgebra.mul!(A, B::AbstractThunk, C) = mul!(A, unthunk(B), C)
LinearAlgebra.mul!(C::AbstractThunk, A, B, α, β) = throw(MutateThunkException())
function LinearAlgebra.mul!(C::AbstractThunk, A::AbstractThunk, B, α, β)
    return throw(MutateThunkException())
end
function LinearAlgebra.mul!(C::AbstractThunk, A, B::AbstractThunk, α, β)
    return throw(MutateThunkException())
end
function LinearAlgebra.mul!(C::AbstractThunk, A::AbstractThunk, B::AbstractThunk, α, β)
    return throw(MutateThunkException())
end
LinearAlgebra.mul!(C, A::AbstractThunk, B, α, β) = mul!(C, unthunk(A), B, α, β)
LinearAlgebra.mul!(C, A, B::AbstractThunk, α, β) = mul!(C, A, unthunk(B), α, β)
function LinearAlgebra.mul!(C, A::AbstractThunk, B::AbstractThunk, α, β)
    return mul!(C, unthunk(A), unthunk(B), α, β)
end

function LinearAlgebra.BLAS.ger!(alpha, x::AbstractThunk, y, A)
    return LinearAlgebra.BLAS.ger!(alpha, unthunk(x), y, A)
end
function LinearAlgebra.BLAS.ger!(alpha, x, y::AbstractThunk, A)
    return LinearAlgebra.BLAS.ger!(alpha, x, unthunk(y), A)
end
function LinearAlgebra.BLAS.gemv!(tA, alpha, A, x::AbstractThunk, beta, y)
    return LinearAlgebra.BLAS.gemv!(tA, alpha, A, unthunk(x), beta, y)
end
function LinearAlgebra.BLAS.gemv(tA, alpha, A, x::AbstractThunk)
    return LinearAlgebra.BLAS.gemv(tA, alpha, A, unthunk(x))
end
function LinearAlgebra.BLAS.scal!(n, a::AbstractThunk, X, incx)
    return LinearAlgebra.BLAS.scal!(n, unthunk(a), X, incx)
end

function LinearAlgebra.LAPACK.trsyl!(transa, transb, A, B, C::AbstractThunk, isgn=1)
    return throw(MutateThunkException())
end

"""
    @thunk expr

Define a [`Thunk`](@ref) wrapping the `expr`, to lazily defer its evaluation.
"""
macro thunk(body)
    # Basically `:(Thunk(() -> $(esc(body))))` but use the location where it is defined.
    # so we get useful stack traces if it errors.
    func = Expr(:->, Expr(:tuple), Expr(:block, __source__, body))
    return :(Thunk($(esc(func))))
end

"""
    unthunk(x)

On `AbstractThunk`s this removes 1 layer of thunking.
On any other type, it is the identity operation.
"""
@inline unthunk(x) = x

Base.conj(x::AbstractThunk) = @thunk(conj(unthunk(x)))
Base.adjoint(x::AbstractThunk) = @thunk(adjoint(unthunk(x)))
Base.transpose(x::AbstractThunk) = @thunk(transpose(unthunk(x)))

#####
##### `Thunk`
#####

"""
    Thunk(()->v)
A thunk is a deferred computation.
It wraps a zero argument closure that when invoked returns a tangent.
`@thunk(v)` is a macro that expands into `Thunk(()->v)`.

To evaluate the wrapped closure, call [`unthunk`](@ref) which is a no-op when the
argument is not a `Thunk`.

```jldoctest
julia> t = @thunk(3)
Thunk(var"#4#5"())

julia> unthunk(t)
3
```

### When to `@thunk`?
When writing `rrule`s (and to a lesser exent `frule`s), it is important to `@thunk`
appropriately.
Propagation rules that return multiple derivatives may not have all deriviatives used.
 By `@thunk`ing the work required for each derivative, they then compute only what is needed.

#### How do thunks prevent work?
If we have `res = pullback(...) = @thunk(f(x)), @thunk(g(x))`
then if we did `dx + res[1]` then only `f(x)` would be evaluated, not `g(x)`.
Also if we did `ZeroTangent() * res[1]` then the result would be `ZeroTangent()` and `f(x)` would not be evaluated.

#### So why not thunk everything?
`@thunk` creates a closure over the expression, which (effectively) creates a `struct`
with a field for each variable used in the expression, and call overloaded.

Do not use `@thunk` if this would be equal or more work than actually evaluating the expression itself.
This is commonly the case for scalar operators.

For more details see the manual section [on using thunks effectively](https://juliadiff.org/ChainRulesCore.jl/dev/rule_author/writing_good_rules.html#Use-Thunks-appropriately).
"""
struct Thunk{F} <: AbstractThunk
    f::F
end

@inline unthunk(x::Thunk) = unthunk(x.f())

function Base.show(io::IO, x::Thunk)
    print(io, "Thunk(")
    str = sprint(show, x.f, context = io)  # often this name is like "ChainRules.var"#1398#1403"{Matrix{Float64}, Matrix{Float64}}"
    ind = findfirst("var\"#", str)
    if isnothing(ind) || length(str) < 80
        printstyled(io, str, color=:light_black)
    else
        printstyled(io, str[1:ind[5]], "...", color=:light_black)
    end
    print(io, ")")
end

Base.convert(::Type{<:Thunk}, a::AbstractZero) = @thunk(a)

"""
    InplaceableThunk(add!::Function, val::Thunk)

A wrapper for a `Thunk`, that allows it to define an inplace `add!` function.

`add!` should be defined such that: `ithunk.add!(Δ) = Δ .+= ithunk.val`
but it should do this more efficently than simply doing this directly.
(Otherwise one can just use a normal `Thunk`).

Most operations on an `InplaceableThunk` treat it just like a normal `Thunk`;
and destroy its inplacability.
"""
struct InplaceableThunk{T<:Thunk,F} <: AbstractThunk
    add!::F
    val::T
end

unthunk(x::InplaceableThunk) = unthunk(x.val)

function Base.show(io::IO, x::InplaceableThunk)
    print(io, "InplaceableThunk(")
    str = sprint(show, x.add!, context = io)
    ind = findfirst("var\"#", str)  # look for auto-generated function names, often with huge types
    if isnothing(ind)
        printstyled(io, str, color=:light_black)
    else
        printstyled(io, str[1:ind[5]], "...", color=:light_black)
    end
    print(io, ", ")
    show(io, x.val)
    print(io, ")")
end


"""
    BroadcastThunk(bc::Broadcasted)

Another kind of thunk, which wraps Base's lazy broadcasting.
Usually constructed by `@bc_thunk expr`.

Calling `unthunk` will materialise the array. But inserting it directly
into a broadcast will fuse the old and the new, which is the whole point.

That means that rules accepting thunks should not always `unthunk(dy)` them
before use. Instead, they may rely on broadcasting to do so, as long as
they only do this in one place. Or they can call `unthunk_or_bc(dy)`
to explicitly remove any other thunk, but keep a lazy `Broadcasted`.
This can be fused into several different broadcasting expressions,
for example in the `rrule` for `x ./ y`.

This possibility of multiple use means that `BroadcastThunk` should
only contain cheap operations.
"""
struct BroadcastThunk{T, B<:Broadcast.Broadcasted} <: AbstractThunk
    bc::B
end
function BroadcastThunk(bc::Broadcast.Broadcasted)
    T = Base.@default_eltype(bc)
    return if T <: Number  # applicable(zero, T)  # will SVector work?
        BroadcastThunk{T, typeof(bc)}(bc)
    else
        # We need init=zero(T) for unbroadcast to work.
        # For things like arrays of arrays ... perhaps make an old-style thunk?
        InplaceableThunk(dx -> dx .+= bc, @thunk copy(bc))
    end
end

Base.eltype(x::BroadcastThunk{T}) where {T} = T

@inline unthunk(x::BroadcastThunk) = copy(x.bc)

# This is the whole point:
Base.Broadcast.broadcastable(x::BroadcastThunk) = x.bc

function Base.show(io::IO, x::BroadcastThunk)
    print(io, "BroadcastThunk{")
    show(io, eltype(x))
    print(io, "}(")
    str = sprint(show, x.bc, context = io)
    if length(str) < 80
        printstyled(io, str, color=:light_black)
    else
        printstyled(io, str[1:70], "...", color=:light_black)
    end
    print(io, ")")
end

"""
    @bc_thunk f(a, g(b, c))

This works like `@.` to produce something like `@thunk f.(a, g.(b, c))`.
Except that instead of a `Thunk`, it's usually a `BroadcastThunk`.
(The exception is that, for broadcasts whose eltype is not `<:Number`,
it will make an `InplaceableThunk` instead.)

Some rules with `@thunk f.(a, g.(b, c))` should not use this!
The resulting broadcast may be done a few times.
"""
macro bc_thunk(ex)
  bc = esc(Broadcast.__dot__(ex))
  :($_lazy_bc.($bc))
end

function _lazy_bc end
Broadcast.broadcasted(::typeof(_lazy_bc), x) = _Lazy_BC(x)
struct _Lazy_BC{T}; bc::T; end
Broadcast.materialize(x::_Lazy_BC) = BroadcastThunk(Broadcast.instantiate(x.bc))

macro bc_thunk(s::Symbol)
    error("cannot apply @bc_thunk to one symbol, there is nothing to broadcast!")
end

# These prove useful for writing rules:

Base.:(-)(x::BroadcastThunk) = @bc_thunk -(x.bc)

for fun in [:conj, :real, :imag, :complex]
    @eval Base.$fun(x::BroadcastThunk) = BroadcastThunk(Broadcast.instantiate(Broadcast.broadcasted($fun, x.bc)))
end

Base.sum(x::BroadcastThunk; dims=:) = _sum_bc(x.bc, dims)
_sum_bc(bc, ::Colon) = sum(bc)
_sum_bc(bc, dims) = sum(bc; dims, init=zero(eltype(x)))

LinearAlgebra.dot(x::Base.AbstractArrayOrBroadcasted, y::BroadcastThunk{<:Number}) = sum(@bc_thunk conj(x) * y.bc)

"""
    unthunk_or_bc(dx)

This removes most thunks, but turns a `BroadcastThunk` into a `Broadcasted`.
For use in rrules which can handle the latter.
"""
unthunk_or_bc(x) = unthunk(x)
unthunk_or_bc(x::BroadcastThunk) = x.bc
unthunk_or_bc(x::Thunk) = unthunk_or_bc(x.f())  # bct = @bc_thunk 1+[2,3]; unthunk_or_bc(@thunk -bct) isa Broadcasted

# The accumulation of thunks is a mess, no AD actually calls add!!, and + always unthunks.
# But... these should be safe, and make more BroadcastThunks:
Base.:(+)(x::BroadcastThunk, y::BroadcastThunk) = @bc_thunk x.bc + y.bc

Base.:(+)(x::BroadcastThunk, y::AbstractArray) = @bc_thunk x.bc + y
Base.:(+)(x::AbstractArray, y::BroadcastThunk) = @bc_thunk x + y.bc

Base.:(+)(x::BroadcastThunk, y::AbstractThunk) = x + unthunk(y)
Base.:(+)(x::AbstractThunk, y::BroadcastThunk) = unthunk(x) + y
