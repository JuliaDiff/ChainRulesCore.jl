"""
    StructuralTangent{P} <: AbstractTangent

Representing the type of the tangent of a `struct` `P` (or a `Tuple`/`NamedTuple`).
as an object with mirroring fields.

!!!!!! warning Exprimental
    `StructuralTangent` is an experimental feature, and is part of the mutation support featureset.
    The `StructuralTangent` constructor returns a `MutableTangent` for mutable structs.
    `MutableTangent` is an experimental feature.
    Thus use of `StructuralTangent` (rather than `Tangent` directly) is also experimental.
    While this notice remains it may have changes in behavour, and interface in any _minor_ version of ChainRulesCore.
"""
abstract type StructuralTangent{P} <: AbstractTangent end

"""
    Tangent{P, T} <: StructuralTangent{P} <: AbstractTangent

This type represents the tangent for a `struct`/`NamedTuple`, or `Tuple`.
`P` is the the corresponding primal type that this is a tangent for.

`Tangent{P}` should have fields (technically properties), that match to a subset of the
fields of the primal type; and each should be a tangent type matching to the primal
type of that field.
Fields of the P that are not present in the Tangent are treated as `Zero`.

`T` is an implementation detail representing the backing data structure.
For Tuple it will be a Tuple, and for everything else it will be a `NamedTuple`.
It should not be passed in by user.

For `Tangent`s of `Tuple`s, `iterate` and `getindex` are overloaded to behave similarly
to for a tuple.
For `Tangent`s of `struct`s, `getproperty` is overloaded to allow for accessing values
via `tangent.fieldname`.
Any fields not explictly present in the `Tangent` are treated as being set to `ZeroTangent()`.
To make a `Tangent` have all the fields of the primal the [`canonicalize`](@ref)
function is provided.
"""
struct Tangent{P,T} <: StructuralTangent{P}
    # Note: If T is a Tuple/Dict, then P is also a Tuple/Dict
    # (but potentially a different one, as it doesn't contain tangents)
    backing::T

    function Tangent{P,T}(backing) where {P,T}
        if P <: Tuple
            T <: Tuple || _backing_error(P, T, Tuple)
        elseif P <: AbstractDict
            T <: AbstractDict || _backing_error(P, T, AbstractDict)
        elseif P === Any  # can be anything
        else  # Any other struct (including NamedTuple)
            T <: NamedTuple || _backing_error(P, T, NamedTuple)
        end
        return new(backing)
    end
end

"""
    MutableTangent{P}(fields) <: StructuralTangent{P} <: AbstractTangent

This type represents the tangent to a mutable struct.
It itself is also mutable.

!!! warning Exprimental
    MutableTangent is an experimental feature, and is part of the mutation support featureset.
    While this notice remains it may have changes in behavour, and interface in any _minor_ version of ChainRulesCore.
    Exactly how it should be used (e.g. is it forward-mode only?)

!!! warning Do not directly mess with the tangent backing data
    It is relatively straight forward for a forwards-mode AD to work correctly in the presence of mutation and aliasing of primal values.
    However, this requires that the tangent is aliased in turn and conversely that it is copied when the primal is).
    If you seperately alias the backing data, etc by using the internal `ChainRulesCore.backing` function you can break this.
"""
struct MutableTangent{P,F} <: StructuralTangent{P}
    backing::F

    function MutableTangent{P}(
        any_mask::NamedTuple{names, <:NTuple{<:Any, Bool}}, fvals::NamedTuple{names}
    ) where {names, P}

        backing = map(any_mask, fvals) do isany, fval
            ref = if isany
                Ref{Any}
            else
                Ref
            end
            return ref(fval)
        end
        return new{P, typeof(backing)}(backing)
    end

    function MutableTangent{P}(fvals) where P
        any_mask = NamedTuple{fieldnames(P)}((!isconcretetype).(fieldtypes(P)))
        return MutableTangent{P}(any_mask, fvals)
    end
end

####################################################################
# StructuralTangent Common


function StructuralTangent{P}(nt::NamedTuple) where {P}
    if has_mutable_tangent(P)
        return MutableTangent{P}(nt)
    else
        return Tangent{P,typeof(nt)}(nt)
    end
end


has_mutable_tangent(::Type{P}) where P = ismutabletype(P) && (!isabstracttype(P) && fieldcount(P) > 0)


StructuralTangent{P}(tup::Tuple) where P = Tangent{P,typeof(tup)}(tup)
StructuralTangent{P}(dict::Dict) where P = Tangent{P}(dict)

Base.keys(tangent::StructuralTangent) = keys(backing(tangent))
Base.propertynames(tangent::StructuralTangent) = propertynames(backing(tangent))

Base.haskey(tangent::StructuralTangent, key) = haskey(backing(tangent), key)
if isdefined(Base, :hasproperty)
    function Base.hasproperty(tangent::StructuralTangent, key::Symbol)
        return hasproperty(backing(tangent), key)
    end
end

Base.iszero(t::StructuralTangent) = all(iszero, backing(t))

function Base.map(f, tangent::StructuralTangent{P}) where {P}
    #TODO: is it even useful to support this on MutableTangents?
    #TODO: we implictly assume only linear `f` are called and that it is safe to ignore noncanonical Zeros
    # This feels like a fair assumption since all normal operations on tangents are linear
    L = propertynames(backing(tangent))
    vals = map(f, Tuple(backing(tangent)))
    named_vals = NamedTuple{L,typeof(vals)}(vals)
    return if tangent isa MutableTangent
        MutableTangent{P}(named_vals)
    else
        Tangent{P,typeof(named_vals)}(named_vals)
    end
end


function Base.show(io::IO, tangent::StructuralTangent{P}) where {P}
    if tangent isa MutableTangent
        print(io, "Mutable")
    end
    print(io, "Tangent{")
    str = sprint(show, P, context = io)
    i = findfirst('{', str)
    if isnothing(i)
        print(io, str)
    else  # for Tangent{T{A,B,C}}(stuff), print {A,B,C} in grey, and trim this part if longer than a line:
        print(io, str[1:prevind(str, i)])
        if length(str) < 80
            printstyled(io, str[i:end], color=:light_black)
        else
           printstyled(io, str[i:prevind(str, 80)], "...", color=:light_black) 
        end
    end
    print(io, "}")
    if isempty(backing(tangent))
        print(io, "()")  # so it doesn't show `NamedTuple()`
    else
        # allow Tuple or NamedTuple `show` to do the rendering of brackets etc
        show(io, backing(tangent))
    end
end

"""
    backing(x)

Accesses the backing field of a `Tangent`,
or destructures any other struct type into a `NamedTuple`.
Identity function on `Tuple`s and `NamedTuple`s.

This is an internal function used to simplify operations between `Tangent`s and the
primal types.
"""
backing(x::Tuple) = x
backing(x::NamedTuple) = x
backing(x::Dict) = x
backing(x::Tangent) = getfield(x, :backing)
backing(x::MutableTangent) = map(getindex, getfield(x, :backing))

# For generic structs
function backing(x::T)::NamedTuple where {T}
    # note: all computation outside the if @generated happens at runtime.
    # so the first 4 lines of the branchs look the same, but can not be moved out.
    # see https://github.com/JuliaLang/julia/issues/34283
    if @generated
        !isstructtype(T) &&
            throw(DomainError(T, "backing can only be used on struct types"))
        nfields = fieldcount(T)
        names = fieldnames(T)
        types = fieldtypes(T)

        vals = Expr(:tuple, ntuple(ii -> :(getfield(x, $ii)), nfields)...)
        return :(NamedTuple{$names,Tuple{$(types...)}}($vals))
    else
        !isstructtype(T) &&
            throw(DomainError(T, "backing can only be used on struct types"))
        nfields = fieldcount(T)
        names = fieldnames(T)
        types = fieldtypes(T)

        vals = ntuple(ii -> getfield(x, ii), nfields)
        return NamedTuple{names,Tuple{types...}}(vals)
    end
end

"""
    _zeroed_backing(P)

Returns a NamedTuple with same fields as `P`, and all values `ZeroTangent()`.
"""
@generated function _zeroed_backing(::Type{P}) where {P}
    nil_base = ntuple(fieldcount(P)) do i
        (fieldname(P, i), ZeroTangent())
    end
    return (; nil_base...)
end

"""
    construct(::Type{T}, fields::[NamedTuple|Tuple])

Constructs an object of type `T`, with the given fields.
Fields must be correct in name and type, and `T` must have a default constructor.

This internally is called to construct structs of the primal type `T`,
after an operation such as the addition of a primal to a tangent

It should be overloaded, if `T` does not have a default constructor,
or if `T` needs to maintain some invarients between its fields.
"""
function construct(::Type{T}, fields::NamedTuple{L}) where {T,L}
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

construct(::Type{T}, fields::T) where {T<:NamedTuple} = fields
construct(::Type{T}, fields::T) where {T<:Tuple} = fields

elementwise_add(a::Tuple, b::Tuple) = map(+, a, b)

function elementwise_add(a::NamedTuple{an}, b::NamedTuple{bn}) where {an,bn}
    # Rule of Tangent addition: any fields not present are implict hard Zeros

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
        return (; vals...)
    end
end

elementwise_add(a::Dict, b::Dict) = merge(+, a, b)

struct PrimalAdditionFailedException{P} <: Exception
    primal::P
    tangent
    original::Exception
end

function Base.showerror(io::IO, err::PrimalAdditionFailedException{P}) where {P}
    println(io, "Could not construct $P after addition.")
    println(io, "This probably means no default constructor is defined.")
    println(io, "Either define a default constructor")
    printstyled(io, "$P(", join(propertynames(err.tangent), ", "), ")"; color=:blue)
    println(io, "\nor overload")
    printstyled(
        io, "ChainRulesCore.construct(::Type{$P}, ::$(typeof(err.tangent)))"; color=:blue
    )
    println(io, "\nor overload")
    printstyled(io, "Base.:+(::$P, ::$(typeof(err.tangent)))"; color=:blue)
    println(io, "\nOriginal Exception:")
    printstyled(io, err.original; color=:yellow)
    return println(io)
end

#######################################
# immutable Tangent

function Tangent{P}(; kwargs...) where {P}
    backing = (; kwargs...)  # construct as NamedTuple
    return Tangent{P,typeof(backing)}(backing)
end

function Tangent{P}(args...) where {P}
    return Tangent{P,typeof(args)}(args)
end

function Tangent{P}() where {P<:Tuple}
    backing = ()
    return Tangent{P,typeof(backing)}(backing)
end

function Tangent{P}(d::Dict) where {P<:AbstractDict}
    return Tangent{P,typeof(d)}(d)
end

function _backing_error(P, G, E)
    msg = "Tangent for the primal $P should be backed by a $E type, not by $G."
    return throw(ArgumentError(msg))
end

function Base.:(==)(a::Tangent{P,T}, b::Tangent{P,T}) where {P,T}
    return backing(a) == backing(b)
end
function Base.:(==)(a::Tangent{P}, b::Tangent{P}) where {P}
    all_fields = union(keys(backing(a)), keys(backing(b)))
    return all(getproperty(a, f) == getproperty(b, f) for f in all_fields)
end
Base.:(==)(a::Tangent{P}, b::Tangent{Q}) where {P,Q} = false

Base.hash(a::Tangent, h::UInt) = Base.hash(backing(canonicalize(a)), h)

Base.iszero(::Tangent{<:,NamedTuple{}}) = true
Base.iszero(::Tangent{<:,Tuple{}}) = true


Base.first(tangent::Tangent{P,T}) where {P,T<:Union{Tuple,NamedTuple}} = first(backing(canonicalize(tangent)))
Base.last(tangent::Tangent{P,T}) where {P,T<:Union{Tuple,NamedTuple}} = last(backing(canonicalize(tangent)))

Base.tail(t::Tangent{P}) where {P<:Tuple} = Tangent{_tailtype(P)}(Base.tail(backing(canonicalize(t)))...)
@generated _tailtype(::Type{P}) where {P<:Tuple} = Tuple{P.parameters[2:end]...}
Base.tail(t::Tangent{<:Tuple{Any}}) = NoTangent()
Base.tail(t::Tangent{<:Tuple{}}) = NoTangent()

Base.tail(t::Tangent{P}) where {P<:NamedTuple} = Tangent{_tailtype(P)}(; Base.tail(backing(canonicalize(t)))...)
_tailtype(::Type{NamedTuple{S,P}}) where {S,P} = NamedTuple{Base.tail(S), _tailtype(P)}
Base.tail(t::Tangent{<:NamedTuple{<:Any, <:Tuple{Any}}}) = NoTangent()
Base.tail(t::Tangent{<:NamedTuple{<:Any, <:Tuple{}}}) = NoTangent()

function Base.getindex(tangent::Tangent{P,T}, idx::Int) where {P,T<:Union{Tuple,NamedTuple}}
    back = backing(canonicalize(tangent))
    return unthunk(getfield(back, idx))
end
function Base.getindex(tangent::Tangent{P,T}, idx::Symbol) where {P,T<:NamedTuple}
    hasfield(T, idx) || return ZeroTangent()
    return unthunk(getfield(backing(tangent), idx))
end
function Base.getindex(tangent::Tangent, idx)
    return unthunk(getindex(backing(tangent), idx))
end

function Base.getproperty(tangent::Tangent, idx::Int)
    back = backing(canonicalize(tangent))
    return unthunk(getfield(back, idx))
end
function Base.getproperty(tangent::Tangent{P,T}, idx::Symbol) where {P,T<:NamedTuple}
    hasfield(T, idx) || return ZeroTangent()
    return unthunk(getfield(backing(tangent), idx))
end


Base.iterate(tangent::Tangent, args...) = iterate(backing(tangent), args...)
Base.length(tangent::Tangent) = length(backing(tangent))

Base.eltype(::Type{<:Tangent{<:Any,T}}) where {T} = eltype(T)
function Base.reverse(tangent::Tangent)
    rev_backing = reverse(backing(tangent))
    return Tangent{typeof(rev_backing),typeof(rev_backing)}(rev_backing)
end

function Base.indexed_iterate(tangent::Tangent{P,<:Tuple}, i::Int, state=1) where {P}
    return Base.indexed_iterate(backing(tangent), i, state)
end

function Base.map(f, tangent::Tangent{P,<:Tuple}) where {P}
    vals::Tuple = map(f, backing(tangent))
    return Tangent{P,typeof(vals)}(vals)
end
function Base.map(f, tangent::Tangent{P,<:Dict}) where {P<:Dict}
    return Tangent{P}(Dict(k => f(v) for (k, v) in backing(tangent)))
end

Base.conj(tangent::Tangent) = map(conj, tangent)



"""
    canonicalize(tangent::Tangent{P}) -> Tangent{P}

Return the canonical `Tangent` for the primal type `P`.
The property names of the returned `Tangent` match the field names of the primal,
and all fields of `P` not present in the input `tangent` are explictly set to `ZeroTangent()`.
"""
function canonicalize(tangent::Tangent{P,<:NamedTuple{L}}) where {P,L}
    nil = _zeroed_backing(P)
    combined = merge(nil, backing(tangent))
    if length(combined) !== fieldcount(P)
        throw(
            ArgumentError(
                "Tangent fields do not match primal fields.\n" *
                "Tangent fields: $L. Primal ($P) fields: $(fieldnames(P))",
            ),
        )
    end
    return Tangent{P,typeof(combined)}(combined)
end

# Tuple tangents are always in their canonical form
canonicalize(tangent::Tangent{<:Tuple,<:Tuple}) = tangent

# Dict tangents are always in their canonical form.
canonicalize(tangent::Tangent{<:Any,<:AbstractDict}) = tangent

# Tangents of unspecified primal types (indicated by specifying exactly `Any`)
# all combinations of type-params are specified here to avoid ambiguities
canonicalize(tangent::Tangent{Any,<:NamedTuple{L}}) where {L} = tangent
canonicalize(tangent::Tangent{Any,<:Tuple}) = tangent
canonicalize(tangent::Tangent{Any,<:AbstractDict}) = tangent

###################################################
# MutableTangent

MutableTangent{P}(;kwargs...) where P = MutableTangent{P}(NamedTuple(kwargs))

ref_backing(t::MutableTangent) = getfield(t, :backing)

Base.getproperty(tangent::MutableTangent, idx::Symbol) = getfield(ref_backing(tangent), idx)[]
Base.getproperty(tangent::MutableTangent, idx::Int) = getfield(ref_backing(tangent), idx)[]  # break ambig

function Base.setproperty!(tangent::MutableTangent, name::Symbol, x)
    return getfield(ref_backing(tangent), name)[] = x
end
function Base.setproperty!(tangent::MutableTangent, idx::Int, x)
    return getfield(ref_backing(tangent), idx)[] = x
end  # break ambig

Base.hash(tangent::MutableTangent, h::UInt) = hash(backing(tangent), h)
function Base.:(==)(t1::MutableTangent{T1}, t2::MutableTangent{T2}) where {T1, T2}
    typeintersect(T1, T2) == Union{} && return false
    backing(t1)==backing(t2)
end
