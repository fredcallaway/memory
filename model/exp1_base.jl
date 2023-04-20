# %% ==================== simulation ====================

function exp1_mdp(prm)
    time_cost = (MS_PER_SAMPLE / 1000) * .1
    MetaMDP{1}(;allow_stop=true, max_step=MAX_STEP, miss_cost=3,
        prm.threshold, prm.noise, sample_cost=prm.sample_cost + time_cost,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end

function simulate_exp1(make_policies::Function, prm::NamedTuple, N=100_000)
    simulate_exp1(make_policies(prm)..., N; prm.between_σ, prm.within_σ, prm.judgement_noise)
end

function simulate_exp1(pre_pol::Policy, crit_pol::Policy, N=100_000; 
                       between_σ, within_σ, judgement_noise)

    strengths = sample_strengths(pre_pol,  N; between_σ, within_σ)
    df = map(strengths) do (strength, pretest_accuracy)
        sim = simulate(crit_pol; s=(strength,), fix_log=RTLog())
        post = posterior(crit_pol.m, sim.b)[1]
        (;
            response_type = sim.b.focused == -1 ? "empty" : "correct",
            rt=sim.fix_log.rt * MS_PER_SAMPLE,
            judgement=post.μ,
            pretest_accuracy,
        )
    end |> DataFrame
    discretize_judgement!(df, judgement_noise)
end

function discretize_judgement!(df, noise)
    df.judgement .+= rand(Normal(0, noise), nrow(df))
    breaks = map(["empty", "correct"]) do rtyp
        human = @subset(human_trials, :response_type .== rtyp).judgement
        model = @subset(df, :response_type .== rtyp).judgement
        target_prop = counts(human) ./ length(human)
        rtyp => quantile(model, cumsum(target_prop)) 
    end |> Dict
    df.judgement = map(df.response_type, df.judgement) do rtyp, j
        i = findfirst(j .≤ breaks[rtyp])
        something(i, length(breaks))  # judgment greater than all breaks (floating point error)
    end
    df
end

# %% ==================== likelihood ====================

function make_hist(trials::DataFrame; dt=MS_PER_SAMPLE, maxt=MAX_TIME)
    @chain trials begin
        @rtransform :rt = quantize(:rt, dt)
        wrap_counts(
            rt=dt:dt:maxt, 
            response_type=["correct", "empty"], 
            # judgement=1:5,  # not fitting judgements
            pretest_accuracy=0:0.5:1
        ) 
        normalize!
    end
end

function compute_histograms(name, make_policies, prms; N=100000, read_only=false, enable_cache=true)
    compute_cached("$(name)_histograms_$N", prms) do prm
        make_hist(simulate_exp1(make_policies, prm, N))
    end
end

function compute_loss(histograms, prms; sort=true)
    tbl = DataFrame(prms)
    results = @showprogress "loss " pmap(histograms, prms) do model_hist, prm
        ismissing(model_hist) && return [Inf, 1000., 1000.]  # 1000 gives super long RT (a warning flag)
        if hasfield(typeof(prm), :α_ndt)
            (;α_ndt, θ_ndt) = prm
            lk = likelihood(model_hist, human_hist, Gamma(α_ndt, θ_ndt))
            [lk, α_ndt, θ_ndt]
        else
            res = optimize_ndt(model_hist, human_hist)
            [res.minimum; res.minimizer]
        end
    end
    tbl.loss, tbl.α_ndt, tbl.θ_ndt = invert(results)
    sort && sort!(tbl, :loss)
    tbl
end

# %% ==================== effect fitting ====================

function compute_effects(name, make_policies, prms; N=100000, read_only=false, enable_cache=true)
    compute_cached("$(name)_effects_$N", prms; read_only, enable_cache) do prm
        exp1_effects(make_policies, prm, N)
    end
end

function exp1_effects(make_policies::Function, prm::NamedTuple, N=100_000)
    sim = simulate_exp1(make_policies, prm, N)
    ndt = Gamma(prm.α_ndt, prm.θ_ndt)
    sim.rt = sim.rt .+ rand(ndt, nrow(sim))
    exp1_effects(sim)
end

function exp1_effects(trials)
    accuracy = mean(trials.response_type .== "correct")
    pretest_accuracy = mean(trials.pretest_accuracy)

    empty_judgement = @chain trials begin
        @rsubset :response_type == "empty"
        @regress rt ~ judgement
        get_coef
    end
    
    empty_pretest = @chain trials begin
        @rsubset :response_type == "empty"
        @regress rt ~ pretest_accuracy
        get_coef
    end

    correct_judgement = @chain trials begin
        @rsubset :response_type == "correct"
        @regress rt ~ -judgement
        get_coef
    end
    
    correct_pretest = @chain trials begin
        @rsubset :response_type == "correct"
        @regress rt ~ -pretest_accuracy
        get_coef
    end

    (;pretest_accuracy, accuracy, empty_judgement, empty_pretest, correct_judgement, correct_pretest)
end


