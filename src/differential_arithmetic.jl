#==
All differentials need to define + and *.
That happens here.

We just use @eval to define all the combinations for AbstractDifferential
subtypes, as we know the full set that might be encountered.
Thus we can avoid any ambiguities.

Notice:
    The precidence goes: (:Wirtinger, :Zero, :DoesNotExist, :One, :AbstractThunk, :Any)
    Thus each of the @eval loops creating definitions of + and *
    defines the combination this type with all types of  lower precidence.
    This means each eval loops is 1 item smaller than the previous.
==#


function Base.:*(a::Wirtinger, b::Wirtinger)
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

function Base.:+(a::Wirtinger, b::Wirtinger)
    return Wirtinger(+(a.primal, b.primal), a.conjugate + b.conjugate)
end

for T in (:Zero, :DoesNotExist, :One, :AbstractThunk, :Any)
    @eval Base.:+(a::Wirtinger, b::$T) = a + Wirtinger(b, Zero())
    @eval Base.:+(a::$T, b::Wirtinger) = Wirtinger(a, Zero()) + b

    @eval Base.:*(a::Wirtinger, b::$T) = Wirtinger(a.primal * b, a.conjugate * b)
    @eval Base.:*(a::$T, b::Wirtinger) = Wirtinger(a * b.primal, a * b.conjugate)
end


Base.:+(::Zero, b::Zero) = Zero()
Base.:*(::Zero, ::Zero) = Zero()
for T in (:DoesNotExist, :One, :AbstractThunk, :Any)
    @eval Base.:+(::Zero, b::$T) = b
    @eval Base.:+(a::$T, ::Zero) = a

    @eval Base.:*(::Zero, ::$T) = Zero()
    @eval Base.:*(::$T, ::Zero) = Zero()
end


Base.:+(::DoesNotExist, ::DoesNotExist) = DoesNotExist()
Base.:*(::DoesNotExist, ::DoesNotExist) = DoesNotExist()
for T in (:One, :AbstractThunk, :Any)
    @eval Base.:+(::DoesNotExist, b::$T) = b
    @eval Base.:+(a::$T, ::DoesNotExist) = a

    @eval Base.:*(::DoesNotExist, ::$T) = DoesNotExist()
    @eval Base.:*(::$T, ::DoesNotExist) = DoesNotExist()
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
