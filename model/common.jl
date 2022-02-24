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
stringify(nt::NamedTuple) = replace(string(map(x->round(x; digits=6), nt::NamedTuple)), ([" ", "(", ")"] .=> "")...)

function sample_strengths(pol, N=10000; strength_drift=Normal(0, .5))
    map(1:N) do i
        s = sample_state(pol.m)
        pretest_accuracy = mean(simulate(pol; s).b.focused == 1 for i in 1:2)
        strength = only(s) + rand(strength_drift)
        (strength, pretest_accuracy)
    end
end

function simulate_pretest(prm, N=10000)
    m = MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=1,
        prm.threshold, prm.sample_cost, prm.noise,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
    pol = OptimalPolicy(m; dv=m.threshold*.02)
    
    mapreduce(vcat, 1:N) do i
        s = sample_state(pol.m)
        map(1:2) do j
            sim = simulate(pol; s, fix_log=RTLog())
            (;
                wid="optimal",
                word=i,
                strength=only(s),
                response_type = sim.b.focused == -1 ? "empty" : "correct",
                rt=sim.fix_log.rt * ms_per_sample,
            )
        end
    end |> DataFrame
end

function pretest_metrics(df)
    acc_rate = @chain df begin
        @by([:word, :wid], :x=mean(:response_type .== "correct"))
        @by(_, :x, :n=length(:x) ./ nrow(_))
        wrapdims(:n, :x)
        sortkeys
    end


    rt_μ, rt_σ = @chain df begin
        @rsubset :response_type .== "correct"
        @with mean(:rt), std(:rt)
    end

    (;acc_rate, rt_μ, rt_σ)
end

pre_prms = grid(10, Box(
    drift_μ = (0, 1),
    drift_σ = (0.5, 1.5),
    threshold = (1, 10),
    sample_cost = (.001, .05, :log),
    noise = (.2, 2),
))
