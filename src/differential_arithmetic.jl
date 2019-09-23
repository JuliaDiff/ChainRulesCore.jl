#==
All differentials need to define + and *.
That happens here.

We just use @eval to define all the combinations for AbstractDifferential
subtypes, as we know the full set that might be encountered.
Thus we can avoid any ambiguities.

Notice:
    The precidence goes: (:AbstractWirtinger, :Casted, :Zero, :DNE, :One, :AbstractThunk, :Any)
    Thus each of the @eval loops creating definitions of + and *
    defines the combination this type with all types of  lower precidence.
    This means each eval loops is 1 item smaller than the previous.
==#


function Base.:*(a::Union{Complex,AbstractWirtinger},
                 b::Union{Complex,AbstractWirtinger})
    error("""
          Cannot multiply two Wirtinger objects; this error likely means a
          `WirtingerRule` was inappropriately defined somewhere. Multiplication
          of two Wirtinger objects is not defined because chain rule application
          often expands into a non-commutative operation in the Wirtinger
          calculus. To put it another way: simply given two Wirtinger objects
          and no other information, we can't know "locally" which components to
          conjugate in order to implement the chain rule. We could pick a
          convention; for example, we could define `a::Wirtinger * b::Wirtinger`
          such that we assume the chain rule application is of the form `f_a ∘ f_b`
          instead of `f_b ∘ f_a`. However, picking such a convention is likely to
          lead to silently incorrect derivatives due to commutativity assumptions
          in downstream generic code that deals with the reals. Thus, ChainRulesCore
          makes this operation an error instead.
          """)
end

function Base.:+(a::AbstractWirtinger, b::AbstractWirtinger)
    return Wirtinger(wirtinger_primal(a) + wirtinger_primal(b),
                     wirtinger_conjugate(a) + wirtinger_conjugate(b))
end

Base.:+(a::ComplexGradient, b::ComplexGradient) = ComplexGradient(a.val + b.val)

for T in (:Casted, :Zero, :DNE, :One, :AbstractThunk)
    @eval Base.:+(a::AbstractWirtinger, b::$T) = a + Wirtinger(b, Zero())
    @eval Base.:+(a::$T, b::AbstractWirtinger) = Wirtinger(a, Zero()) + b

    @eval Base.:*(a::Wirtinger, b::$T) = Wirtinger(a.primal * b, a.conjugate * b)
    @eval Base.:*(a::$T, b::Wirtinger) = Wirtinger(a * b.primal, a * b.conjugate)

    @eval Base.:*(a::ComplexGradient, b::$T) = ComplexGradient(a.val * b)
    @eval Base.:*(a::$T, b::ComplexGradient) = ComplexGradient(a * b.val)
end

Base.:+(a::AbstractWirtinger, b) = a + Wirtinger(b, Zero())
Base.:+(a, b::AbstractWirtinger) = Wirtinger(a, Zero()) + b

Base.:*(a::Wirtinger, b::Real) = Wirtinger(a.primal * b, a.conjugate * b)
Base.:*(a::Real, b::Wirtinger) = Wirtinger(a * b.primal, a * b.conjugate)

Base.:*(a::ComplexGradient, b::Real) = ComplexGradient(a.val * b)
Base.:*(a::Real, b::ComplexGradient) = ComplexGradient(a * b.val)


Base.:+(a::Casted, b::Casted) = Casted(broadcasted(+, a.value, b.value))
Base.:*(a::Casted, b::Casted) = Casted(broadcasted(*, a.value, b.value))
for T in (:Zero, :DNE, :One, :AbstractThunk, :Any)
    @eval Base.:+(a::Casted, b::$T) = Casted(broadcasted(+, a.value, b))
    @eval Base.:+(a::$T, b::Casted) = Casted(broadcasted(+, a, b.value))

    @eval Base.:*(a::Casted, b::$T) = Casted(broadcasted(*, a.value, b))
    @eval Base.:*(a::$T, b::Casted) = Casted(broadcasted(*, a, b.value))
end


Base.:+(::Zero, b::Zero) = Zero()
Base.:*(::Zero, ::Zero) = Zero()
for T in (:DNE, :One, :AbstractThunk, :Any)
    @eval Base.:+(::Zero, b::$T) = b
    @eval Base.:+(a::$T, ::Zero) = a

    @eval Base.:*(::Zero, ::$T) = Zero()
    @eval Base.:*(::$T, ::Zero) = Zero()
end


Base.:+(::DNE, ::DNE) = DNE()
Base.:*(::DNE, ::DNE) = DNE()
for T in (:One, :AbstractThunk, :Any)
    @eval Base.:+(::DNE, b::$T) = b
    @eval Base.:+(a::$T, ::DNE) = a

    @eval Base.:*(::DNE, ::$T) = DNE()
    @eval Base.:*(::$T, ::DNE) = DNE()
end


Base.:+(a::One, b::One) = extern(a) + extern(b)
Base.:*(::One, ::One) = One()
for T in (:AbstractThunk, :Any)
    @eval Base.:+(a::One, b::$T) = extern(a) + b
    @eval Base.:+(a::$T, b::One) = a + extern(b)

    @eval Base.:*(::One, b::$T) = b
    @eval Base.:*(a::$T, ::One) = a
end


Base.:+(a::AbstractThunk, b::AbstractThunk) = extern(a) + extern(b)
Base.:*(a::AbstractThunk, b::AbstractThunk) = extern(a) * extern(b)
for T in (:Any,)
    @eval Base.:+(a::AbstractThunk, b::$T) = extern(a) + b
    @eval Base.:+(a::$T, b::AbstractThunk) = a + extern(b)

    @eval Base.:*(a::AbstractThunk, b::$T) = extern(a) * b
    @eval Base.:*(a::$T, b::AbstractThunk) = a * extern(b)
end

@inline function chain(outer, inner, swap_order=false)
    if swap_order
        return Wirtinger(
                         wirtinger_primal(inner) * wirtinger_primal(outer) +
                         conj(wirtinger_conjugate(inner)) * wirtinger_conjugate(outer),
                         wirtinger_conjugate(inner) * wirtinger_primal(outer) +
                         conj(wirtinger_primal(inner) * wirtinger_conjugate(outer))
                        ) |> refine_differential
    end
    return Wirtinger(
                     wirtinger_primal(outer) * wirtinger_primal(inner) +
                     wirtinger_conjugate(outer) * conj(wirtinger_conjugate(inner)),
                     wirtinger_primal(outer) * wirtinger_conjugate(inner) +
                     wirtinger_conjugate(outer) * conj(wirtinger_primal(inner))
                    ) |> refine_differential
end

@inline function chain(outer::ComplexGradient, inner, swap_order=false)
    if swap_order
        return ComplexGradient(
                               (wirtinger_conjugate(inner) + conj(wirtinger_primal(inner))) * 
                               outer.val
                              )
    end
    return ComplexGradient(
                           outer.val *
                           (wirtinger_conjugate(inner) + conj(wirtinger_primal(inner)))
                          )
end

@inline function chain(outer::ComplexGradient, inner::ComplexGradient, swap_order=false)
    if swap_order
        return ComplexGradient(conj(inner.val) * outer.val)
    end
    return ComplexGradient(outer.val * conj(inner.val))
end
