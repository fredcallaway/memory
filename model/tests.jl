using Test

@testset "parse_fixation" begin
    @test parse_fixations([1,1,1,2,2]) == [3, 2]
    @test parse_fixations([1]) == [1]
    @test parse_fixations([1, 1]) == [2]
    @test parse_fixations([]) == []
end

#model1 = BackwardsInduction(N=2,max_step=30, threshold=60, sample_cost=0, miss_cost=1)
#compute_value_functions!(model1)
#@assert 1 - pdf(BetaBinomial(60, 1, 1), 60) ≈ -model1.V[1,1,1,1,1]