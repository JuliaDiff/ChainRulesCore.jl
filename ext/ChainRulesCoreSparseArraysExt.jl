module ChainRulesCoreSparseArraysExt

using ChainRulesCore
using ChainRulesCore: project_type, _projection_mismatch
using SparseArrays: SparseVector, SparseMatrixCSC, nzrange, rowvals

ChainRulesCore.is_inplaceable_destination(::SparseVector) = true
ChainRulesCore.is_inplaceable_destination(::SparseMatrixCSC) = true

# Word from on high is that we should regard all un-stored values of sparse arrays as
# structural zeros. Thus ProjectTo needs to store nzind, and get only those.
# This implementation very naiive, can probably be made more efficient.

function ChainRulesCore.ProjectTo(x::SparseVector{T}) where {T<:Number}
    return ProjectTo{SparseVector}(;
        element=ProjectTo(zero(T)), nzind=x.nzind, axes=axes(x)
    )
end
function (project::ProjectTo{SparseVector})(dx::AbstractArray)
    dy = if axes(dx) == project.axes
        dx
    else
        if size(dx, 1) != length(project.axes[1])
            throw(_projection_mismatch(project.axes, size(dx)))
        end
        reshape(dx, project.axes)
    end
    nzval = map(i -> project.element(dy[i]), project.nzind)
    return SparseVector(length(dx), project.nzind, nzval)
end
function (project::ProjectTo{SparseVector})(dx::SparseVector)
    if size(dx) != map(length, project.axes)
        throw(_projection_mismatch(project.axes, size(dx)))
    end
    # When sparsity pattern is unchanged, all the time is in checking this,
    # perhaps some simple hash/checksum might be good enough?
    samepattern = project.nzind == dx.nzind
    # samepattern = length(project.nzind) == length(dx.nzind)
    if eltype(dx) <: project_type(project.element) && samepattern
        return dx
    elseif samepattern
        nzval = map(project.element, dx.nzval)
        SparseVector(length(dx), dx.nzind, nzval)
    else
        nzind = project.nzind
        # Or should we intersect? Can this exploit sorting?
        # nzind = intersect(project.nzind, dx.nzind)
        nzval = map(i -> project.element(dx[i]), nzind)
        return SparseVector(length(dx), nzind, nzval)
    end
end

function ChainRulesCore.ProjectTo(x::SparseMatrixCSC{T}) where {T<:Number}
    return ProjectTo{SparseMatrixCSC}(;
        element=ProjectTo(zero(T)),
        axes=axes(x),
        rowval=rowvals(x),
        nzranges=nzrange.(Ref(x), axes(x, 2)),
        colptr=x.colptr,
    )
end
# You need not really store nzranges, you can get them from colptr -- TODO
# nzrange(S::AbstractSparseMatrixCSC, col::Integer) = getcolptr(S)[col]:(getcolptr(S)[col+1]-1)
function (project::ProjectTo{SparseMatrixCSC})(dx::AbstractArray)
    dy = if axes(dx) == project.axes
        dx
    else
        if size(dx) != (length(project.axes[1]), length(project.axes[2]))
            throw(_projection_mismatch(project.axes, size(dx)))
        end
        reshape(dx, project.axes)
    end
    nzval = Vector{project_type(project.element)}(undef, length(project.rowval))
    k = 0
    for col in project.axes[2]
        for i in project.nzranges[col]
            row = project.rowval[i]
            val = dy[row, col]
            nzval[k += 1] = project.element(val)
        end
    end
    m, n = map(length, project.axes)
    return SparseMatrixCSC(m, n, project.colptr, project.rowval, nzval)
end

function (project::ProjectTo{SparseMatrixCSC})(dx::SparseMatrixCSC)
    if size(dx) != map(length, project.axes)
        throw(_projection_mismatch(project.axes, size(dx)))
    end
    samepattern = dx.colptr == project.colptr && dx.rowval == project.rowval
    # samepattern = length(dx.colptr) == length(project.colptr) && dx.colptr[end] == project.colptr[end]
    if eltype(dx) <: project_type(project.element) && samepattern
        return dx
    elseif samepattern
        nzval = map(project.element, dx.nzval)
        m, n = size(dx)
        return SparseMatrixCSC(m, n, dx.colptr, dx.rowval, nzval)
    else
        invoke(project, Tuple{AbstractArray}, dx)
    end
end

end # module
