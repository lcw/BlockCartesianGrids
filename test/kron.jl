@testset "Eye" begin
    @test size(Eye{Bool,3}()) == (3, 3)
    @test eltype(Eye{Bool,7}()) == Bool
    @test I(6) == Eye{Bool,6}()
end

@testset "Kron" begin
    rng = StableRNG(37)

    @time for T in (Float64, Double64)
        a = rand(rng, T, 3, 2)
        b = rand(rng, T, 4, 5)
        c = rand(rng, T, 1, 7)

        for args in (
            (a,),
            (a, Eye{T,5}()),
            (Eye{T,2}(), b),
            (a, b),
            (Eye{T,3}(), Eye{T,2}(), c),
            (Eye{T,2}(), b, Eye{T,7}()),
            (a, Eye{T,4}(), Eye{T,7}()),
            (a, b, c)
        )

            K = collect(Kron(args))

            is = reverse(size.(args, 2))
            js = 2:(length(args)+1)

            d = rand(T, is..., js...)
            @test vec(Kron(args) * d) ≈ vec(K * reshape(d, prod(is), prod(js)),)

            d = rand(SVector{2,T}, is..., js...)
            @test vec(Kron(args) * d) ≈ vec(K * reshape(d, prod(is), prod(js)),)
        end
    end
end