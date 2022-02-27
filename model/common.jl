include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("figure.jl")
include("box.jl")

using DataFrames, DataFramesMeta, CSV
using ProgressMeter

ms_per_sample = 200

squared(x) = x^2
load_data(name) = CSV.read("../data/processed/$name.csv", DataFrame, missingstring="NA")
stringify(nt::NamedTuple) = replace(string(map(x->round(x; digits=8), nt::NamedTuple)), ([" ", "(", ")"] .=> "")...)

macro bywrap(x, what, val)
    arg = :(:_val = $val)
    esc(quote
        b = $(DataFramesMeta.by_helper(x, what, arg))
        what_ = $what isa Symbol ? ($what,) : $what
        wrapdims(b, :_val, what_..., sort=true)
    end)
end

function sample_strengths(pol, N=10000; strength_drift=Normal(0, .5))
    map(1:N) do i
        s = sample_state(pol.m)
        pretest_accuracy = mean(simulate(pol; s).b.focused == 1 for i in 1:2)
        strength = only(s) + rand(strength_drift)
        (strength, pretest_accuracy)
    end
end

function pretest_mdp(prm)
    time_cost = (ms_per_sample / 1000) * (.25 / 15)
    MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=1,
        prm.threshold, prm.noise, sample_cost=prm.sample_cost + time_cost,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end
