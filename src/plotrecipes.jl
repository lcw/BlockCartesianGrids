RecipesBase.@recipe function f(grid::AbstractBlockGrid{T,2}) where {T}
    xs, = skeleton(grid)

    xlims --> (xs[begin], xs[end])
    legend --> :none
    grid --> false

    # Block lines
    RecipesBase.@series begin
        seriestype --> :scatterpath
        marker --> :vline
        linecolor --> :grey
        linewidth --> 1

        xs, fill(zero(eltype(xs)), size(xs))
    end

    # Grid points
    RecipesBase.@series begin
        seriestype --> :scatter
        x = vec(getindex.(grid, 1))
        x, fill(zero(eltype(x)), size(x))
    end
end

RecipesBase.@recipe function f(grid::AbstractBlockGrid{T,4}) where {T}
    xs, ys = skeleton(grid)

    xlims --> (xs[begin], xs[end])
    ylims --> (ys[begin], ys[end])
    aspect_ratio --> :equal
    legend --> false
    grid --> false

    # Block lines
    RecipesBase.@series begin
        seriestype --> :path
        linecolor --> :grey
        linewidth --> 1

        lines = Tuple{Real,Real}[]
        for y in ys
            append!(lines, ((xs[begin], y), (xs[end], y), (NaN, NaN)))
        end
        for x in xs
            append!(lines, ((x, ys[begin]), (x, ys[end]), (NaN, NaN)))
        end

        lines
    end

    # Grid points
    RecipesBase.@series begin
        seriestype --> :scatter
        x = vec(getindex.(grid, 1))
        y = vec(getindex.(grid, 2))
        x, y
    end
end

RecipesBase.@recipe function f(grid::AbstractBlockGrid{T,6}) where {T}
    xs, ys, zs = skeleton(grid)

    xlims --> (xs[begin], xs[end])
    ylims --> (ys[begin], ys[end])
    zlims --> (zs[begin], zs[end])
    aspect_ratio --> :equal
    legend --> false
    grid --> false

    # Block lines
    RecipesBase.@series begin
        seriestype --> :path
        linecolor --> :grey
        linewidth --> 1

        lines = Tuple{Real,Real,Real}[]
        for y in ys, z in zs
            append!(lines, ((xs[begin], y, z), (xs[end], y, z), (NaN, NaN, NaN)))
        end
        for x in xs, z in zs
            append!(lines, ((x, ys[begin], z), (x, ys[end], z), (NaN, NaN, NaN)))
        end
        for x in xs, y in ys
            append!(lines, ((x, y, zs[begin]), (x, y, zs[end]), (NaN, NaN, NaN)))
        end

        lines
    end

    # Grid points
    RecipesBase.@series begin
        seriestype --> :scatter
        x = vec(getindex.(grid, 1))
        y = vec(getindex.(grid, 2))
        z = vec(getindex.(grid, 3))
        x, y, z
    end
end

RecipesBase.@recipe function f(grid::AbstractBlockGrid{T,2}, data::AbstractArray) where {T}
    xs, = skeleton(grid)
    Z = reshape(data, size(grid))

    xlims --> (xs[begin], xs[end])

    for i = axes(grid, 2)
        RecipesBase.@series begin
            seriestype --> :path
            x = getindex.(grid[:, i], 1)
            x, Z[:, i]
        end
    end
end

RecipesBase.@recipe function f(grid::AbstractBlockGrid{T,4}, data::AbstractArray) where {T}
    xs, ys = skeleton(grid)
    Z = reshape(data, size(grid))

    xlims --> (xs[begin], xs[end])
    ylims --> (ys[begin], ys[end])
    aspect_ratio --> :equal
    colorbar --> true

    for i = axes(grid, 3), j = axes(grid, 4)
        RecipesBase.@series begin
            seriestype --> :heatmap
            x = getindex.(grid[:, begin, i, j], 1)
            y = getindex.(grid[begin, :, i, j], 2)
            x, y, reverse(rotr90(Z[:, :, i, j]), dims=2)
        end
    end
end