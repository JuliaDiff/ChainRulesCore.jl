"""
    (p::ProjectTo{T})(dx)

Projects the differential `dx` onto a specific tangent space.

The type `T` is meant to encode the largest acceptable space, so usually
this enforces `p(dx)::T`. But some subspaces which aren't subtypes of `T` may
be allowed, and in particular `dx::AbstractZero` always passes through.

Usually `T` is the "outermost" part of the type, and `p` stores additional 
properties such as projectors for each constituent field.
Arrays have either one projector `p.element` expressing the element type for 
an array of numbers, or else an array of projectors `p.elements`.
These properties can be supplied as keyword arguments on construction,
`p = ProjectTo{T}(; field=data, element=Projector(x))`. For each `T` in use,
corresponding methods should be written for `ProjectTo{T}(dx)` with nonzero `dx`.

When called on `dx::Thunk`, the projection is inserted into the thunk.
"""
struct ProjectTo{P,D<:NamedTuple}
    info::D
end
ProjectTo{P}(info::D) where {P,D<:NamedTuple} = ProjectTo{P,D}(info)
ProjectTo{P}(; kwargs...) where {P} = ProjectTo{P}(NamedTuple(kwargs))

Base.getproperty(p::ProjectTo, name::Symbol) = getproperty(backing(p), name)
Base.propertynames(p::ProjectTo) = propertynames(backing(p))
backing(project::ProjectTo) = getfield(project, :info)

project_type(p::ProjectTo{T}) where {T} = T

function Base.show(io::IO, project::ProjectTo{T}) where {T}
    print(io, "ProjectTo{")
    show(io, T)
    print(io, "}")
    if isempty(backing(project))
        print(io, "()")
    else
        show(io, backing(project))
    end
end

# Structs
# Generic method is to recursively make `ProjectTo`s for all their fields. Not actually 
# used on unknown structs, but useful for handling many known ones in the same manner.
function generic_projector(x::T; kw...) where {T}
    fields_nt::NamedTuple = backing(x)
    fields_proj = map(_maybe_projector, fields_nt)
    # We can't use `T` because if we have `Foo{Matrix{E}}` it should be allowed to make a
    # `Foo{Diagaonal{E}}` etc. We assume it has a default constructor that has all fields 
    # but if it doesn't `construct` will give a good error message.
    wrapT = T.name.wrapper
    # Official API for this? https://github.com/JuliaLang/julia/issues/35543
    return ProjectTo{wrapT}(; fields_proj..., kw...)
end

function generic_projection(project::ProjectTo{T}, dx::T) where {T}
    sub_projects = backing(project)
    sub_dxs = backing(dx)
    return construct(T, map(_maybe_call, sub_projects, sub_dxs))
end

function (project::ProjectTo{T})(dx::Tangent) where {T}
    sub_projects = backing(project)
    sub_dxs = backing(canonicalize(dx))
    return construct(T, map(_maybe_call, sub_projects, sub_dxs))
end

# Used for encoding fields, leaves alone non-diff types:
_maybe_projector(x::Union{AbstractArray,Number,Ref}) = ProjectTo(x)
_maybe_projector(x) = x
# Used for re-constructing fields, restores non-diff types:
_maybe_call(f::ProjectTo, x) = f(x)
_maybe_call(f, x) = f

"""
    ProjectTo(x)

Returns a `ProjectTo{T}` functor which projects a differential `dx` onto the
relevant tangent space for `x`.

At present this undersands only `x::Number`, `x::AbstractArray` and `x::Ref`.
It should not be called on arguments of an `rrule` method which accepts other types.

# Examples
```jldoctest
julia> pr = ProjectTo(1.5f0)  # preserves real numbers, and floating point precision
ProjectTo{Float32}()

julia> pr(3 + 4im)
3.0f0

julia> pd = ProjectTo(Diagonal([1,2,3]))  # preserves structured matrices
ProjectTo{Diagonal}(diag = ProjectTo{AbstractArray}(element = ProjectTo{Float64}(), axes = (Base.OneTo(3),)),)

julia> th = @thunk reshape(1:9,3,3);

julia> pd(th) isa Thunk
true

julia> unthunk(pd(th))
3×3 Diagonal{Float64, Vector{Float64}}:
 1.0   ⋅    ⋅ 
  ⋅   5.0   ⋅ 
  ⋅    ⋅   9.0

julia> ProjectTo([1 2; 3 4]')  # no special structure, integers are promoted to float(x)
ProjectTo{AbstractArray}(element = ProjectTo{Float64}(), axes = (Base.OneTo(2), Base.OneTo(2)))
```
"""
ProjectTo(::Any) # just to attach docstring

# Generic
(::ProjectTo{T})(dx::T) where {T} = dx  # not always correct but we have special cases for when it isn't
(::ProjectTo{T})(dx::AbstractZero) where {T} = dx
(::ProjectTo{T})(dx::NotImplemented) where {T} = dx

# Thunks
(project::ProjectTo)(dx::Thunk) = Thunk(project ∘ dx.f)

# Zero
ProjectTo(::AbstractZero) = ProjectTo{NoTangent}()  # Any x::Zero in forward pass makes this one projector,
(::ProjectTo{NoTangent})(dx) = NoTangent()          # but this is the projection only for nonzero gradients,
(::ProjectTo{NoTangent})(::NoTangent) = NoTangent() # and this one solves an ambiguity.
