using Test

@testset "parse_fixation" begin
    @test parse_fixations([1,1,1,2,2]) == [3, 2]
    @test parse_fixations([1]) == [1]
    @test parse_fixations([1, 1]) == [2]
    @test parse_fixations([]) == []
end