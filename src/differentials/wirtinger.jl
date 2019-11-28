"""
    Wirtinger(primal::Union{Number,AbstractDifferential},
              conjugate::Union{Number,AbstractDifferential})

Returns a `Wirtinger` instance representing the complex differential:

```
df = ∂f/∂z * dz + ∂f/∂z̄ * dz̄
```

where `primal` corresponds to `∂f/∂z * dz` and `conjugate` corresponds to `∂f/∂z̄ * dz̄`.

The two fields of the returned instance can be accessed generically via the
[`wirtinger_primal`](@ref) and [`wirtinger_conjugate`](@ref) methods.
"""
struct Wirtinger{P,C} <: AbstractDifferential
    primal::P
    conjugate::C
    function Wirtinger(primal::Union{Number,AbstractDifferential},
                       conjugate::Union{Number,AbstractDifferential})
        return new{typeof(primal),typeof(conjugate)}(primal, conjugate)
    end
end

wirtinger_primal(x::Wirtinger) = x.primal
wirtinger_primal(x) = x

wirtinger_conjugate(x::Wirtinger) = x.conjugate
wirtinger_conjugate(::Any) = Zero()

extern(x::Wirtinger) = throw(ArgumentError("`Wirtinger` cannot be converted to an external type."))

Base.Broadcast.broadcastable(w::Wirtinger) = Wirtinger(broadcastable(w.primal),
                                                       broadcastable(w.conjugate))

Base.iterate(x::Wirtinger) = (x, nothing)
Base.iterate(::Wirtinger, ::Any) = nothing

# TODO: define `conj` for` `Wirtinger`
Base.conj(x::Wirtinger) = throw(MethodError(conj, x))

function refine_differential(::Type{<:Union{<:Real, AbstractArray{<:Real}}}, w::Wirtinger)
    return wirtinger_primal(w) + wirtinger_conjugate(w)
end
