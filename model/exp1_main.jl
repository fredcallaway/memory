include("common.jl")

trials = CSV.read("../data/processed/exp1/trials.csv", DataFrame)
mkpath("results/exp1")

# %% ==================== Simulate ====================

function discretize_judgement!(df)
    df.judgement .+= rand(Normal(0, 1), nrow(df))
    breaks = map(["empty", "correct"]) do rtyp
        human = @subset(trials, :response_type .== rtyp).judgement
        model = @subset(df, :response_type .== rtyp).judgement
        target_prop = counts(human) ./ length(human)
        rtyp => quantile(model, cumsum(target_prop)) 
    end |> Dict
    df.judgement = map(df.response_type, df.judgement) do rtyp, j
        findfirst(j .≤ breaks[rtyp])
    end
    df
end

function make_frame(pol, N=10000)
    df = map(sample_strengths(pol; N)) do (strength, pretest_accuracy)
        sim = simulate(pol; s=(strength,), fix_log=RTLog())
        post = posterior(pol.m, sim.b)[1]
        (;
            response_type = sim.b.focused == -1 ? "empty" : "correct",
            rt=sim.fix_log.rt * ms_per_sample,
            judgement=post.μ,  # discretized below
            pretest_accuracy,
        )
    end |> DataFrame
    discretize_judgement!(df)
end

m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.06, 
    threshold=7, noise=1.5, max_step=60, prior=Normal(0, 1)
)
ms_per_sample = 200
opt_pol = OptimalPolicy(m)
df = make_frame(opt_pol)
df |> CSV.write("results/exp1/optimal_trials.csv")

rt = @subset(trials, :response_type .== "empty").rt
rand_pol = StopDistributionPolicy2(m, fit(Gamma, rt ./ ms_per_sample))
make_frame(rand_pol) |> CSV.write("results/exp1/random_trials.csv")
