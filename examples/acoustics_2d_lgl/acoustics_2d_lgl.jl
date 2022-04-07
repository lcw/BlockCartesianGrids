using BlockCartesianGrids
using OneDimensionalNodes
using StaticArrays
using StructArrays
using Plots
using LinearAlgebra
using OrdinaryDiffEq

# Add contract to floating point llvm instructions to get more fma instructions
import Base: +, *, -, /
for (jlf, f) in zip((:+, :*, :-, :/), (:add, :mul, :sub, :div))
    for (T, llvmT) in ((Float32, "float"), (Float64, "double"))
        ir = """
            %x = f$f contract nsz $llvmT %0, %1
            ret $llvmT %x
        """
        @eval begin
            # the @pure is necessary so that we can constant propagate.
            Base.@pure function $jlf(a::$T, b::$T)
                Base.@_inline_meta
                Base.llvmcall($ir, $T, Tuple{$T,$T}, a, b)
            end
        end
    end
    @eval function $jlf(args...)
        Base.$jlf(args...)
    end
end

"""
    referencecell(N, FT=Float64)

    Generate the one-dimensional nodes and operators of polynomial order `N`
"""
function referencecell(N, FT=Float64)
    M = N + 1

    # Use `BigFloat` to get correct rounding of points and operators
    r, w = legendregausslobatto(BigFloat, M)

    # Scale for the unit interval
    @. r = (1 + r) / 2
    @. w /= 2

    w⁻¹ = inv.(w)
    D = spectralderivative(r)

    r, w, w⁻¹ = convert.(SVector{M,FT}, (r, w, w⁻¹))
    D = convert(SMatrix{M,M,FT}, D)

    return (; r, w, w⁻¹, D)
end

function referencecell(N::NTuple{2}, FT=Float64)

    r₁, w₁, w₁⁻¹, D₁ = referencecell(N[1], FT)
    r₂, w₂, w₂⁻¹, D₂ =
        (N[1] == N[2]) ? (r₁, w₁, w₁⁻¹, D₁) : referencecell(N[2], FT)

    # Setup arrays for broadcasting
    r₂, w₂, w₂⁻¹ = r₂', w₂', w₂⁻¹'

    return (; r₁, r₂, w₁, w₂, w₁⁻¹, w₂⁻¹, D₁, D₂)
end

"""
    exactsolution(x, t, k=1)

`k`th mode in `[-1,1] × [-1, 1]`` with `n⋅v == 0`` on the boundary.
"""
function exactsolution(x, t, k=one(eltype(x)))
    T = eltype(x)
    w = sqrt(T(2))

    a = (x[1] + 1) * k * π / 2
    b = (x[2] + 1) * k * π / 2
    c = w * k * t * π / 2

    pr = cos(a) * cos(b) * cos(c) * w
    v₁ = sin(a) * cos(b) * sin(c)
    v₂ = cos(a) * sin(b) * sin(c)

    return SVector(pr, v₁, v₂)
end

@inline flux(u, ::Val{1}) = @inbounds SVector(u[2], u[1], -zero(eltype(u)))
@inline flux(u, ::Val{2}) = @inbounds SVector(u[3], -zero(eltype(u)), u[1])
@inline flux(u) = hcat(flux(u, Val(1)), flux(u, Val(2)))

@inline function bc(n, u⁻, u⁺)
    pr, v... = u⁻
    vₙ = n ⋅ v
    SVector(pr, (v - 2vₙ * n)...)
end

@inline function numericalflux(n, u⁻, u⁺)
    pr_jump, v_jump... = u⁺ - u⁻
    pr_ave, v_ave... = (u⁺ + u⁻) / 2

    vₙ_jump = n ⋅ v_jump
    vₙ_ave = n ⋅ v_ave

    SVector(vₙ_ave - pr_jump / 2, ((pr_ave - vₙ_jump / 2) * n)...)
end

@generated function kernel!(du, u, parameters, ::Val{Np₁}, ::Val{Np₂}, ::Val{E₁}, ::Val{E₂}, ::Val{Ns}) where {Np₁,Np₂,E₁,E₂,Ns}
    quote
        D₁, D₂, r₁x₁, r₂x₂, Fscale₁, Fscale₂ = parameters

        @inbounds FT = eltype(u[1])

        f₁r₁ = -zero(MVector{$Ns,FT})
        f₂r₂ = -zero(MVector{$Ns,FT})
        duij = -zero(MVector{$Ns,FT})

        @inbounds for e₂ in 1:$E₂, e₁ in 1:$E₁
            r₁x₁e₁ = r₁x₁[e₁]
            r₂x₂e₂ = r₂x₂[e₂]

            # Volume
            for j in 1:$Np₂
                @simd ivdep for i in 1:$Np₁
                    fill!(f₁r₁, -zero(FT))
                    fill!(f₂r₂, -zero(FT))

                    Base.Cartesian.@nexprs $Np₁ k -> begin
                        f₁kj = flux(u[k, j, e₁, e₂], Val(1))

                        Base.Cartesian.@nexprs $Ns s -> begin
                            f₁r₁[s] += D₁[i, k] * f₁kj[s]
                        end
                    end

                    Base.Cartesian.@nexprs $Np₂ k -> begin
                        f₂ik = flux(u[i, k, e₁, e₂], Val(2))

                        Base.Cartesian.@nexprs $Ns s -> begin
                            f₂r₂[s] += D₂[j, k] * f₂ik[s]
                        end
                    end

                    Base.Cartesian.@nexprs $Ns s -> begin
                        duij[s] = r₁x₁e₁ * f₁r₁[s] + r₂x₂e₂ * f₂r₂[s]
                    end

                    # The following is causing a bounds check when `du` is a `StructArray`
                    du[i, j, e₁, e₂] = -duij
                end
            end

            e₁⁻ = e₁
            e₂⁻ = e₂

            # Face 1
            e₁⁺ = mod1(e₁⁻ - 1, $E₁)
            e₂⁺ = e₂⁻
            n = SVector{2,FT}(-one(FT), -zero(FT))
            Fscale = Fscale₁[e₁]
            Base.Cartesian.@nexprs $Np₂ j -> begin
                u⁻ = u[1, j, e₁⁻, e₂⁻]
                u⁺ = u[end, j, e₁⁺, e₂⁺]
                u⁺ = ifelse(e₁⁻ != 1, u⁺, bc(n, u⁻, u⁺))
                nf = numericalflux(n, u⁻, u⁺)
                f⁻ = flux(u⁻) * n
                du[1, j, e₁⁻, e₂⁻] -= Fscale * (nf - f⁻)
            end

            # Face 2
            e₁⁺ = mod1(e₁⁻ + 1, $E₁)
            e₂⁺ = e₂⁻
            n = SVector{2,FT}(one(FT), -zero(FT))
            Fscale = Fscale₁[e₁]
            Base.Cartesian.@nexprs $Np₂ j -> begin
                u⁻ = u[end, j, e₁⁻, e₂⁻]
                u⁺ = u[1, j, e₁⁺, e₂⁺]
                u⁺ = ifelse(e₁⁻ != $E₁, u⁺, bc(n, u⁻, u⁺))
                nf = numericalflux(n, u⁻, u⁺)
                f⁻ = flux(u⁻) * n
                du[end, j, e₁⁻, e₂⁻] -= Fscale * (nf - f⁻)
            end

            # Face 3
            e₁⁺ = e₁⁻
            e₂⁺ = mod1(e₂⁻ - 1, $E₂)
            n = SVector{2,FT}(-zero(FT), -one(FT))
            Fscale = Fscale₂[e₂]
            Base.Cartesian.@nexprs $Np₁ i -> begin
                u⁻ = u[i, 1, e₁⁻, e₂⁻]
                u⁺ = u[i, end, e₁⁺, e₂⁺]
                u⁺ = ifelse(e₂⁻ != 1, u⁺, bc(n, u⁻, u⁺))
                nf = numericalflux(n, u⁻, u⁺)
                f⁻ = flux(u⁻) * n
                du[i, 1, e₁⁻, e₂⁻] -= Fscale * (nf - f⁻)
            end

            # Face 4
            e₁⁺ = e₁⁻
            e₂⁺ = mod1(e₂⁻ + 1, $E₂)
            n = SVector{2,FT}(-zero(FT), one(FT))
            Fscale = Fscale₂[e₂]
            Base.Cartesian.@nexprs $Np₁ i -> begin
                u⁻ = u[i, end, e₁⁻, e₂⁻]
                u⁺ = u[i, 1, e₁⁺, e₂⁺]
                u⁺ = ifelse(e₂⁻ != $E₂, u⁺, bc(n, u⁻, u⁺))
                nf = numericalflux(n, u⁻, u⁺)
                f⁻ = flux(u⁻) * n
                du[i, end, e₁⁻, e₂⁻] -= Fscale * (nf - f⁻)
            end
        end
        return
    end
end

function rhs!(du, u, parameters, t)
    Np₁ = size(u, 1)
    Np₂ = size(u, 2)
    E₁ = size(u, 3)
    E₂ = size(u, 4)
    Ns = length(eltype(u))
    kernel!(du, u, parameters, Val(Np₁), Val(Np₂), Val(E₁), Val(E₂), Val(Ns))
end

N = (5, 5)
E = (10, 10)
FT = Float64

r₁, r₂, w₁, w₂, w₁⁻¹, w₂⁻¹, D₁, D₂ = referencecell(N, FT)
g = blockgrid((r₁, r₂), (range(-one(FT), one(FT), first(E) + 1), range(-one(FT), one(FT), last(E) + 1)))
r₁x₁, r₂x₂, J₁, J₂ = metrics(g)
Fscale₁ = inv.(first(w₁) .* J₁[:])
Fscale₂ = inv.(first(w₂) .* J₂[:])

tspan = (FT(0), FT(2))
u = StructArray(exactsolution.(g, first(tspan)))
parameters = D₁, D₂, r₁x₁, r₂x₂, Fscale₁, Fscale₂

prob = ODEProblem(rhs!, u, tspan, parameters)
sol = solve(prob, RDPK3SpFSAL35(), reltol=1e-7, abstol=1e-7, save_everystep=false)

ū = StructArray(exactsolution.(g, last(tspan)))
p̄r, v̄₁, v̄₂ = StructArrays.components(ū)
pr, v₁, v₂ = StructArrays.components(sol.u[end])

@show sqrt(
    sum(@. w₁ * w₂ * J₁ * J₂ * ((v₁ - v̄₁)^2 / 2 + (v₂ - v̄₂)^2 / 2 + (pr - p̄r)^2 / 2)),
)

# plot(g, pr)