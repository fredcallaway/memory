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

function simulate_exp1(pre_pol::Policy, crit_pol::Policy, N=100000; strength_drift=Normal(0, 1e-9))
    strengths = sample_strengths(pre_pol,  N; strength_drift)
    df = map(strengths) do (strength, pretest_accuracy)
        sim = simulate(crit_pol; s=(strength,), fix_log=RTLog())
        post = posterior(crit_pol.m, sim.b)[1]
        (;
            response_type = sim.b.focused == -1 ? "empty" : "correct",
            rt=sim.fix_log.rt * ms_per_sample,
            judgement=post.μ,
            pretest_accuracy,
        )
    end |> DataFrame
    discretize_judgement!(df)
end

function pretest_mdp(prm)
    time_cost = (ms_per_sample / 1000) * (.25 / 15)
    MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=1,
        prm.threshold, prm.noise, sample_cost=prm.sample_cost + time_cost,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end

function exp1_mdp(prm)
    time_cost = (ms_per_sample / 1000) * .1
    MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=3,
        prm.threshold, prm.noise, sample_cost=prm.sample_cost + time_cost,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end

function simulate_exp1(prm::NamedTuple, N=100000)
    simulate_exp1(
        OptimalPolicy(pretest_mdp(prm)), 
        OptimalPolicy(exp1_mdp(prm)), 
        N; strength_drift=Normal(prm.strength_drift_μ, prm.strength_drift_σ)
    )
end
