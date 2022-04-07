struct BlockGrid{T,D,B,S} <: AbstractBlockGrid{T,D}
    dims::Dims{D}
    block::B
    skeleton::S
    function BlockGrid(block, skeleton)
        B = typeof(block)
        S = typeof(skeleton)
        if length(block) != length(skeleton)
            throw(ArgumentError("$block and $skeleton need to be the same length"))
        end

        for b in block
            bmin, bmax = extrema(b)
            if bmin < zero(bmin) || bmax > one(bmax)
                throw(ArgumentError("block grid points $b need to be between zero and one"))
            end
        end

        dims = (ntuple(b -> length(block[b]), length(block))...,
            ntuple(s -> length(skeleton[s]) - 1, length(skeleton))...)

        ET = promote_type(ntuple(b -> eltype(block[b]), length(block))...,
            ntuple(s -> eltype(skeleton[s]), length(skeleton))...)

        D = length(block) + length(skeleton)
        T = SVector{D ÷ 2,ET}
        new{T,D,B,S}(dims, block, skeleton)
    end
end

Base.size(grid::BlockGrid) = grid.dims
Base.eltype(::BlockGrid{T}) where {T} = T

@inline Base.@propagate_inbounds function Base.getindex(
    grid::BlockGrid{T,D,B,S},
    i::Vararg{Int,D}
) where {T,D,B,S}
    @boundscheck checkbounds(grid, i...)
    return T(ntuple(D ÷ 2) do d
        @inbounds bd = grid.block[d][@inbounds(i[d])]
        @inbounds sd = grid.skeleton[d][@inbounds(i[d+D÷2])]
        @inbounds se = grid.skeleton[d][@inbounds(i[d+D÷2] + 1)]

        return sd + bd * (se - sd)
    end)
end

skeleton(grid::BlockGrid) = grid.skeleton
block(grid::BlockGrid) = grid.block
function metrics(grid::BlockGrid)
    D = length(skeleton(grid))
    diffs = diff.(skeleton(grid))
    Js = ntuple(i -> reshape(diffs[i], ntuple(j -> j == i + D ? length(diffs[i]) : 1, 2D)), D)
    rxs = ntuple(i -> inv.(Js[i]), D)
    return (rxs..., Js...)
end

blockgrid(block, skeleton) = BlockGrid(block, skeleton)