const RealDifferential = Union{Real,AbstractZero,One}

struct OneForm{T} <: AbstractDifferential
    x::T
end

one_form(x::RealDifferential) = x
one_form(x::Complex) = OneForm(x)
one_form(x::AbstractArray) = OneForm(x)
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

_match_ones(inds::NTuple{N}, ::Val{N}) where {N} = inds
function _match_ones(inds::NTuple{M,I}, ::Val{N})::NTuple{N,I} where {M,N,I}
    if M < N
        return (inds..., ntuple(_ -> one(I), N - M)...)
    else
        for i in N+1:M
            isone(inds[i]) || throw(ArgumentError("trailing indices have to be one"))
        end
        return ntuple(i -> inds[i], N)
    end
end

Base.getindex(a::OneForm, i) = one_form(a.x[i])
function Base.getindex(a::OneForm, inds::Vararg{Any,N}) where {N}
    _inds = reverse(_match_ones(inds, Val(N)))
    return one_form(a.x[_inds...])
end

inner(a::RealDifferential, b::RealDifferential) = a * b
inner(a::Number, b::Number) = real(a) * real(b) + imag(a) * imag(b)
inner(a, b) = sum(i -> inner(a[i], b[i]), eachindex(a, b))
inner(a::AbstractArray{<:Real}, b::AbstractArray{<:Real}) = dot(a, b)

Base.@pure function intersect_names(an::Tuple{Vararg{Symbol}}, bn::Tuple{Vararg{Symbol}})
    names = Symbol[]
    for n in bn
        if sym_in(n, an)
            push!(names, n)
        end
    end
    (names...,)
end

function _inner(a::NamedTuple{an}, b::NamedTuple{bn}) where {an, bn}
    # Rule of Composite addition: any fields not present are implict hard Zeros

    # Base on the `merge(:;NamedTuple, ::NamedTuple)` code from Base.
    # https://github.com/JuliaLang/julia/blob/592748adb25301a45bd6edef3ac0a93eed069852/base/namedtuple.jl#L220-L231
    if @generated
        names = intersect_names(an, bn)
        vals = map(names) do field
            a_field = :(getproperty(a, $(QuoteNode(field))))
            b_field = :(getproperty(b, $(QuoteNode(field))))
            return :(inner($a_field, $b_field))
        end
        return :(+($(vals...)))
    else
        names = intersect_names(an, bn)
        return sum(names) do field
            a_field = getproperty(a, field)
            b_field = getproperty(b, field)
            return inner(a_field, b_field)
        end
    end
end
