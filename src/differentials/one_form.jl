struct OneForm{T} <: AbstractDifferential
    x::T
end

one_form(x::Union{Real,AbstractZero,One}) = x
one_form(x::Complex) = OneForm(x)
one_form(x::AbstractVector) = OneForm(x)
one_form(x::Composite) = OneForm(x)

const ♭ = one_form
♯(x::OneForm) = x.x

function Base.show(io::IO, ::MIME"text/plain", x::T) where {T<:OneForm{<:Complex}}
    z = ♯(x)
    index = get(io, :index, "")

    show(io, T)
    print(io, ":\n ")
    show(io, real(z))
    print(io, " dℜ(z", index, ") + ")
    show(io, imag(z))
    print(io, " dℑ(z", index, ")")
end
