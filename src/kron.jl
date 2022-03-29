struct Eye{T,N} <: AbstractArray{T,2} end
Base.size(::Eye{T,N}) where {T,N} = (N, N)
Base.eltype(::Eye{T}) where {T} = T

@inline Base.@propagate_inbounds function Base.getindex(
    eye::Eye{T,N},
    i::Vararg{Int,2}
) where {T,N}
    @boundscheck checkbounds(eye, i...)
    return @inbounds i[1] == i[2] ? one(T) : zero(T)
end

struct Kron{T}
    args::T
    Kron(args::Tuple) = new{typeof(args)}(args)
end
components(K::Kron) = K.args
Base.collect(K::Kron) = collect(length(K.args) == 1 ? first(K.args) : kron(K.args...))
Base.size(K::Kron, j::Int) = prod(size.(K.args, j))
Base.:(==)(J::Kron, K::Kron) = all(J.args .== K.args)

function Base.:*(K::Kron{Tuple{D}}, f::AbstractArray{T,2}) where {D<:AbstractMatrix,T}
    if reverse(size.(components(K), 2)) != size(f)[1:length(components(K))]
        throw(DimensionMismatch("Kron K has component dimensions $(size.(components(K))), array f has dimensions $(size(f))"))
    end

    (d,) = components(K)

    g = reshape(f, size(d, 2), :)
    r = similar(f, size(d, 1), size(g, 2))

    fill!(r, -zero(T))
    @inbounds for e in axes(g, 2), i in axes(d, 1), l in axes(d, 2)
        r[i, e] += d[i, l] * g[l, e]
    end

    return reshape(r, size(d, 1), size(f, 2))
end

function Base.:*(K::Kron{Tuple{E,D}}, f::AbstractArray{T,4}) where {D<:AbstractMatrix,E<:Eye,T}
    if reverse(size.(components(K), 2)) != size(f)[1:length(components(K))]
        throw(DimensionMismatch("Kron K has component dimensions $(size.(components(K))), array f has dimensions $(size(f))"))
    end

    e₁, d = components(K)

    g = reshape(f, size(d, 2), size(e₁, 2), :)
    r = similar(f, size(d, 1), size(e₁, 1), size(g, 3))

    fill!(r, -zero(T))
    @inbounds for e in axes(g, 3), j in axes(g, 2), i in axes(d, 1), l in axes(d, 2)
        r[i, j, e] += d[i, l] * g[l, j, e]
    end

    return reshape(r, size(d, 1), size(e₁, 1), size(f, 3), size(f, 4))
end

function Base.:*(K::Kron{Tuple{D,E}}, f::AbstractArray{T,4}) where {D<:AbstractMatrix,E<:Eye,T}
    if reverse(size.(components(K), 2)) != size(f)[1:length(components(K))]
        throw(DimensionMismatch("Kron K has component dimensions $(size.(components(K))), array f has dimensions $(size(f))"))
    end

    d, e₂ = components(K)

    g = reshape(f, size(e₂, 2), size(d, 2), :)
    r = similar(f, size(e₂, 1), size(d, 1), size(g, 3))

    fill!(r, -zero(T))
    @inbounds for e in axes(g, 3), l in axes(d, 2), j in axes(d, 1), i in axes(g, 1)
        r[i, j, e] += d[j, l] * g[i, l, e]
    end

    return reshape(r, size(e₂, 1), size(d, 1), size(f, 3), size(f, 4))
end

function Base.:*(K::Kron{Tuple{B,A}}, f::AbstractArray{T,4}) where {A<:AbstractMatrix,B<:AbstractMatrix,T}
    if reverse(size.(components(K), 2)) != size(f)[1:length(components(K))]
        throw(DimensionMismatch("Kron K has component dimensions $(size.(components(K))), array f has dimensions $(size(f))"))
    end
    b, a = components(K)

    #@assert axes(a, 2) == axes(f, 1) && axes(b, 2) == axes(f, 2)

    g = reshape(f, size(a, 2), size(b, 2), :)
    r = similar(f, size(a, 1), size(b, 1), size(g, 3))

    fill!(r, -zero(T))
    @inbounds for e in axes(g, 3), m in axes(b, 2), l in axes(a, 2), j in axes(b, 1), i in axes(a, 1)
        r[i, j, e] += b[j, m] * a[i, l] * g[l, m, e]
    end

    return reshape(r, size(a, 1), size(b, 1), size(f, 3), size(f, 4))
end

function Base.:*(K::Kron{Tuple{E₃,E₂,D}}, f::AbstractArray{T,6}) where {D<:AbstractMatrix,E₃<:Eye,E₂<:Eye,T}
    if reverse(size.(components(K), 2)) != size(f)[1:length(components(K))]
        throw(DimensionMismatch("Kron K has component dimensions $(size.(components(K))), array f has dimensions $(size(f))"))
    end

    e₃, e₂, d = components(K)

    g = reshape(f, size(d, 2), size(e₂, 2), size(e₃, 2), :)
    r = similar(f, size(d, 1), size(e₂, 1), size(e₃, 1), size(g, 4))

    fill!(r, -zero(T))
    @inbounds for e in axes(g, 4), k in axes(e₃, 1), j in axes(e₂, 1), l in axes(d, 2), i in axes(d, 1)
        r[i, j, k, e] += d[i, l] * g[l, j, k, e]
    end

    return reshape(r, size(d, 1), size(e₂, 1), size(e₃, 1), size(f, 4), size(f, 5), size(f, 6))
end

function Base.:*(K::Kron{Tuple{E₃,D,E₁}}, f::AbstractArray{T,6}) where {D<:AbstractMatrix,E₁<:Eye,E₃<:Eye,T}
    if reverse(size.(components(K), 2)) != size(f)[1:length(components(K))]
        throw(DimensionMismatch("Kron K has component dimensions $(size.(components(K))), array f has dimensions $(size(f))"))
    end

    e₃, d, e₁ = components(K)

    g = reshape(f, size(e₁, 2), size(d, 2), size(e₃, 2), :)
    r = similar(f, size(e₁, 1), size(d, 1), size(e₃, 1), size(g, 4))

    fill!(r, -zero(T))
    @inbounds for e in axes(g, 4), k in axes(e₃, 1), l in axes(d, 2), j in axes(d, 1), i in axes(e₁, 1)
        r[i, j, k, e] += d[j, l] * g[i, l, k, e]
    end

    return reshape(r, size(e₁, 1), size(d, 1), size(e₃, 1), size(f, 4), size(f, 5), size(f, 6))
end

function Base.:*(K::Kron{Tuple{D,E₂,E₁}}, f::AbstractArray{T,6}) where {D<:AbstractMatrix,E₁<:Eye,E₂<:Eye,T}
    if reverse(size.(components(K), 2)) != size(f)[1:length(components(K))]
        throw(DimensionMismatch("Kron K has component dimensions $(size.(components(K))), array f has dimensions $(size(f))"))
    end

    d, e₂, e₁ = components(K)

    g = reshape(f, size(e₁, 2), size(e₂, 2), size(d, 2), :)
    r = similar(f, size(e₁, 1), size(e₂, 1), size(d, 1), size(g, 4))

    fill!(r, -zero(T))
    @inbounds for e in axes(g, 4), l in axes(d, 2), k in axes(d, 1), j in axes(e₂, 1), i in axes(e₁, 1)
        r[i, j, k, e] += d[k, l] * g[i, j, l, e]
    end

    return reshape(r, size(e₁, 1), size(e₂, 1), size(d, 1), size(f, 4), size(f, 5), size(f, 6))
end

function Base.:*(K::Kron{Tuple{C,B,A}}, f::AbstractArray{T,6}) where {A<:AbstractMatrix,B<:AbstractMatrix,C<:AbstractMatrix,T}
    if reverse(size.(components(K), 2)) != size(f)[1:length(components(K))]
        throw(DimensionMismatch("Kron K has component dimensions $(size.(components(K))), array f has dimensions $(size(f))"))
    end

    c, b, a = components(K)

    g = reshape(f, size(a, 2), size(b, 2), size(c, 2), :)
    r = similar(f, size(a, 1), size(b, 1), size(c, 1), size(g, 4))

    fill!(r, -zero(T))
    @inbounds for e in axes(g, 4), n in axes(c, 2), m in axes(b, 2), l in axes(a, 2), k in axes(c, 1), j in axes(b, 1), i in axes(a, 1)
        r[i, j, k, e] += c[k, n] * b[j, m] * a[i, l] * g[l, m, n, e]
    end

    return reshape(r, size(a, 1), size(b, 1), size(c, 1), size(f, 4), size(f, 5), size(f, 6))
end