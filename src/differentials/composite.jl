"""
    Composite{P, T} <: AbstractDifferential

This type represents the differential for a `struct`/`NamedTuple`, or `Tuple`.
`P` is the the corresponding primal type that this is a differential for.

`Composite{P}` should have fields (technically properties), that match to a subset of the
fields of the primal type; and each should be a differential type matching to the primal
type of that field.
Fields of the P that are not present in the Composite are treated as `Zero`.

`T` is an implementation detail representing the backing data structure.
For Tuple it will be a Tuple, and for everything else it will be a `NamedTuple`.
It should not be passed in by user.

For `Composite`s of `Tuple`s, `iterate` and `getindex` are overloaded to behave similarly
to for a tuple.
For `Composite`s of `struct`s, `getproperty` is overloaded to allow for accessing values
via `comp.fieldname`.
Any fields not explictly present in the `Composite` are treated as being set to `Zero()`.
To make a `Composite` have all the fields of the primal the [`canonicalize`](@ref)
function is provided.
"""
struct Composite{P, T} <: AbstractDifferential
    # Note: If T is a Tuple/Dict, then P is also a Tuple/Dict
    # (but potentially a different one, as it doesn't contain differentials)
    backing::T
end

function Composite{P}(; kwargs...) where P
    backing = (; kwargs...)  # construct as NamedTuple
    return Composite{P, typeof(backing)}(backing)
end

function Composite{P}(args...) where P
    return Composite{P, typeof(args)}(args)
end

function Composite{P}() where P<:Tuple
    backing = ()
    return Composite{P, typeof(backing)}(backing)
end

function Composite{P}(d::Dict) where {P<:Dict}
    return Composite{P, typeof(d)}(d)
end

function Base.:(==)(a::Composite{P, T}, b::Composite{P, T}) where {P, T}
    return backing(a) == backing(b)
end
function Base.:(==)(a::Composite{P}, b::Composite{P}) where {P, T}
    all_fields = union(keys(backing(a)), keys(backing(b)))
    return all(getproperty(a, f) == getproperty(b, f) for f in all_fields)
end
Base.:(==)(a::Composite{P}, b::Composite{Q}) where {P, Q} = false

Base.hash(a::Composite, h::UInt) = Base.hash(backing(canonicalize(a)), h)

function Base.show(io::IO, comp::Composite{P}) where P
    print(io, "Composite{")
    show(io, P)
    print(io, "}")
    # allow Tuple or NamedTuple `show` to do the rendering of brackets etc
    show(io, backing(comp))
end

Base.convert(::Type{<:NamedTuple}, comp::Composite{<:Any, <:NamedTuple}) = backing(comp)
Base.convert(::Type{<:Tuple}, comp::Composite{<:Any, <:Tuple}) = backing(comp)
Base.convert(::Type{<:Dict}, comp::Composite{<:Dict, <:Dict}) = backing(comp)

Base.getindex(comp::Composite, idx) = getindex(backing(comp), idx)

# for Tuple
Base.getproperty(comp::Composite, idx::Int) = unthunk(getproperty(backing(comp), idx))
function Base.getproperty(
    comp::Composite{P, T}, idx::Symbol
) where {P, T<:NamedTuple}
    hasfield(T, idx) || return Zero()
    return unthunk(getproperty(backing(comp), idx))
end

Base.keys(comp::Composite) = keys(backing(comp))
Base.propertynames(comp::Composite) = propertynames(backing(comp))

Base.haskey(comp::Composite, key) = haskey(backing(comp), key)
if isdefined(Base, :hasproperty)
    Base.hasproperty(comp::Composite, key::Symbol) = hasproperty(backing(comp), key)
end

Base.iterate(comp::Composite, args...) = iterate(backing(comp), args...)
Base.length(comp::Composite) = length(backing(comp))
Base.eltype(::Type{<:Composite{<:Any, T}}) where T = eltype(T)
function Base.reverse(comp::Composite)
    rev_backing = reverse(backing(comp))
    Composite{typeof(rev_backing), typeof(rev_backing)}(rev_backing)
end

function Base.indexed_iterate(comp::Composite{P,<:Tuple}, i::Int, state=1) where {P}
    return Base.indexed_iterate(backing(comp), i, state)
end

function Base.map(f, comp::Composite{P, <:Tuple}) where P
    vals::Tuple = map(f, backing(comp))
    return Composite{P, typeof(vals)}(vals)
end
function Base.map(f, comp::Composite{P, <:NamedTuple{L}}) where{P, L}
    vals = map(f, Tuple(backing(comp)))
    named_vals = NamedTuple{L, typeof(vals)}(vals)
    return Composite{P, typeof(named_vals)}(named_vals)
end
function Base.map(f, comp::Composite{P, <:Dict}) where {P<:Dict}
    return Composite{P}(Dict(k => f(v) for (k, v) in backing(comp)))
end

Base.conj(comp::Composite) = map(conj, comp)

extern(comp::Composite) = backing(map(extern, comp))  # gives a NamedTuple or Tuple


"""
    backing(x)

Accesses the backing field of a `Composite`,
or destructures any other composite type into a `NamedTuple`.
Identity function on `Tuple`. and `NamedTuple`s.

This is an internal function used to simplify operations between `Composite`s and the
primal types.
"""
backing(x::Tuple) = x
backing(x::NamedTuple) = x
backing(x::Dict) = x
backing(x::Composite) = getfield(x, :backing)

function backing(x::T)::NamedTuple where T
    # note: all computation outside the if @generated happens at runtime.
    # so the first 4 lines of the branchs look the same, but can not be moved out.
    # see https://github.com/JuliaLang/julia/issues/34283
    if @generated
        !isstructtype(T) && throw(DomainError(T, "backing can only be use on composite types"))
        nfields = fieldcount(T)
        names = fieldnames(T)
        types = fieldtypes(T)

        vals = Expr(:tuple, ntuple(ii->:(getfield(x, $ii)), nfields)...)
        return :(NamedTuple{$names, Tuple{$(types...)}}($vals))
    else
        !isstructtype(T) && throw(DomainError(T, "backing can only be use on composite types"))
        nfields = fieldcount(T)
        names = fieldnames(T)
        types = fieldtypes(T)

        vals = ntuple(ii->getfield(x, ii), nfields)
        return NamedTuple{names, Tuple{types...}}(vals)
    end
end

"""
    canonicalize(comp::Composite{P}) -> Composite{P}

Return the canonical `Composite` for the primal type `P`.
The property names of the returned `Composite` match the field names of the primal,
and all fields of `P` not present in the input `comp` are explictly set to `Zero()`.
"""
function canonicalize(comp::Composite{P, <:NamedTuple{L}}) where {P,L}
    nil = _zeroed_backing(P)
    combined = merge(nil, backing(comp))
    if length(combined) !== fieldcount(P)
        throw(ArgumentError(
            "Composite fields do not match primal fields.\n" *
            "Composite fields: $L. Primal ($P) fields: $(fieldnames(P))"
        ))
    end
    return Composite{P, typeof(combined)}(combined)
end

# Tuple composites are always in their canonical form
canonicalize(comp::Composite{<:Tuple, <:Tuple}) = comp

# Dict composite are always in their canonical form.
canonicalize(comp::Composite{<:Any, <:AbstractDict}) = comp

"""
    _zeroed_backing(P)

Returns a NamedTuple with same fields as `P`, and all values `Zero()`.
"""
@generated function _zeroed_backing(::Type{P}) where P
    nil_base = ntuple(fieldcount(P)) do i
        (fieldname(P, i), Zero())
    end
    return (; nil_base...)
end

"""
    construct(::Type{T}, fields::[NamedTuple|Tuple])

Constructs an object of type `T`, with the given fields.
Fields must be correct in name and type, and `T` must have a default constructor.

This internally is called to construct structs of the primal type `T`,
after an operation such as the addition of a primal to a composite.

It should be overloaded, if `T` does not have a default constructor,
or if `T` needs to maintain some invarients between its fields.
"""
function construct(::Type{T}, fields::NamedTuple{L}) where {T, L}
    # Tested and verified that that this avoids a ton of allocations
    if length(L) !== fieldcount(T)
        # if length is equal but names differ then we will catch that below anyway.
        throw(ArgumentError("Unmatched fields. Type: $(fieldnames(T)),  NamedTuple: $L"))
    end

    if @generated
        vals = (:(getproperty(fields, $(QuoteNode(fname)))) for fname in fieldnames(T))
        return :(T($(vals...)))
    else
        return T((getproperty(fields, fname) for fname in fieldnames(T))...)
    end
end

construct(::Type{T}, fields::T) where T<:NamedTuple = fields
construct(::Type{T}, fields::T) where T<:Tuple = fields

elementwise_add(a::Tuple, b::Tuple) = map(+, a, b)

function elementwise_add(a::NamedTuple{an}, b::NamedTuple{bn}) where {an, bn}
    # Rule of Composite addition: any fields not present are implict hard Zeros

    # Base on the `merge(:;NamedTuple, ::NamedTuple)` code from Base.
    # https://github.com/JuliaLang/julia/blob/592748adb25301a45bd6edef3ac0a93eed069852/base/namedtuple.jl#L220-L231
    if @generated
        names = Base.merge_names(an, bn)

        vals = map(names) do field
            a_field = :(getproperty(a, $(QuoteNode(field))))
            b_field = :(getproperty(b, $(QuoteNode(field))))
            value_expr = if Base.sym_in(field, an)
                if Base.sym_in(field, bn)
                    # in both
                    :($a_field + $b_field)
                else
                    # only in `an`
                    a_field
                end
            else # must be in `b` only
                b_field
            end
            Expr(:kw, field, value_expr)
        end
        return Expr(:tuple, Expr(:parameters, vals...))
    else
        names = Base.merge_names(an, bn)
        vals = map(names) do field
            value = if Base.sym_in(field, an)
                a_field = getproperty(a, field)
                if Base.sym_in(field, bn)
                    # in both
                    b_field = getproperty(b, field)
                    a_field + b_field
                else
                    # only in `an`
                    a_field
                end
            else # must be in `b` only
                getproperty(b, field)
            end
            field => value
        end
        return (;vals...)
    end
end

elementwise_add(a::Dict, b::Dict) = merge(+, a, b)

struct PrimalAdditionFailedException{P} <: Exception
    primal::P
    differential::Composite{P}
    original::Exception
end

function Base.showerror(io::IO, err::PrimalAdditionFailedException{P}) where {P}
    println(io, "Could not construct $P after addition.")
    println(io, "This probably means no default constructor is defined.")
    println(io, "Either define a default constructor")
    printstyled(io, "$P(", join(propertynames(err.differential), ", "), ")", color=:blue)
    println(io, "\nor overload")
    printstyled(io,
        "ChainRulesCore.construct(::Type{$P}, ::$(typeof(err.differential)))";
        color=:blue
    )
    println(io, "\nor overload")
    printstyled(io, "Base.:+(::$P, ::$(typeof(err.differential)))"; color=:blue)
    println(io, "\nOriginal Exception:")
    printstyled(io, err.original; color=:yellow)
    println(io)
end

"""
    NO_FIELDS

Constant for the reverse-mode derivative with respect to a structure that has no fields.
The most notable use for this is for the reverse-mode derivative with respect to the
function itself, when that function is not a closure.
"""
const NO_FIELDS = Zero()
