module BlockCartesianGrids

using RecipesBase
using StaticArrays

"""
    AbstractBlockGrid{T,B,C,D} <: AbstractArray{T,D}
Abstract type for representing a blocked grid of of `div(D,2)`-dimensional
blocks in a `div(D,2)`-dimensional grid.  The full grid is a `D`-dimensional.
The grid `g` is a lazy construction of the vector of grid point coordinates
where `eltype(g) = T isa SVector{D}`.
"""
abstract type AbstractBlockGrid{T,D} <: AbstractArray{T,D} end

export Eye, Kron
export blockgrid, metrics, skeleton

include("blockgrid.jl")
include("kron.jl")
include("plotrecipes.jl")

end # module