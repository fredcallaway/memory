include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("figure.jl")
using DataFrames, DataFramesMeta, CSV
using ProgressMeter

ms_per_sample = 200

function sample_strengths(pol, N=10000; strength_drift=Normal(0, .5))
    map(1:N) do i
        s = sample_state(pol.m)
        pretest_accuracy = mean(simulate(pol; s).b.focused == 1 for i in 1:2)
        strength = only(s) + rand(strength_drift)
        (strength, pretest_accuracy)
    end
end