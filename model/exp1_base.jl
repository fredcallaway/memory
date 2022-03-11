function discretize_judgement!(df, noise)
    df.judgement .+= rand(Normal(0, noise), nrow(df))
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

function simulate_exp1(pre_pol::Policy, crit_pol::Policy, N=100000; 
                       strength_drift=Normal(0, 1e-9), judgement_noise=0.)

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
    discretize_judgement!(df, judgement_noise)
end

function exp1_mdp(prm)
    time_cost = (ms_per_sample / 1000) * .1
    MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=3,
        prm.threshold, prm.noise, sample_cost=prm.sample_cost + time_cost,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end

function simulate_exp1(make_policies::Function, prm::NamedTuple, N=100000)
    strength_drift = Normal(prm.strength_drift_μ, prm.strength_drift_σ)
    simulate_exp1(make_policies(prm)..., N; strength_drift, prm.judgement_noise)
end


# %% ==================== summary statistics ====================

function unroll_trial!(P, rt, response_type; dt)
    max_step = size(P, 1)
    n_step = round(Int, rt / dt)
    P[1:n_step, 1] .+= 1 
    outcome = response_type == "correct" ? 2 : 3
    P[n_step+1:max_step, outcome] .+= 1
end

function unroll_time(trials; dt=ms_per_sample, maxt=15000)
    @chain trials begin
        groupby(:pretest_accuracy)
        combine(_) do d
            P = zeros(Int(maxt/dt), 3)
            for t in eachrow(d)
                unroll_trial!(P, t.rt, t.response_type; dt)
            end
            P ./= nrow(d)
            Ref(P)  # prevents unrolling the array
        end
        @orderby :pretest_accuracy
        @with combinedims(:x1)
        KeyedArray( 
            time=dt:dt:maxt, 
            event=[:thinking, :recalled, :skipped], 
            pretest_accuracy=0:0.5:1
        )
    end 
end

function exp1_sumstats(trials)
    X = KeyedArray(zeros(25,2,5), rt=200:200:5000, response_type=["correct", "empty"], judgement=1:5)
    for t in eachrow(trials)
        rt = min(Int(cld(t.rt, 200)), 25)
        rtype = Int(t.response_type == "empty") + 1
        X[rt, rtype, t.judgement] += 1        
    end
    normalize(X)
    # try
    #     # rt = @chain trials begin
    #     #     groupby([:response_type, :pretest_accuracy, :judgement])
    #     #     @combine begin
    #     #         :μ = mean(:rt)
    #     #         :σ = std(:rt)
    #     #         :n = length(:rt)
    #     #     end
    #     # end
    #     @chain trials begin
    #         @rtransform :rt = 200 * cld(:rt, 200)
    #         @bywrap [:rt, :response_type, :judgement, :pretest_accuracy] length(:rt) 0
    #     end
    #     # (;rt, hist, unrolled = unroll_time(trials)
    # catch
    #     missing
    # end
end
