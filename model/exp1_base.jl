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
        findfirst(j .≤ breaks[rtyp])
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
            judgement=1:5,
            pretest_accuracy=0:0.5:1
        ) 
        normalize!
    end
end

function compute_histograms(name, make_policies, prms; N=100000, read_only = false)
    dir = "cache/$(RUN)_exp1_$(name)_histograms_$N"
    mkpath(dir)
    map = read_only ? asyncmap : pmap
    @showprogress "histograms " map(prms) do prm
        cache("$dir/$(hash(prm))"; read_only) do
            try
                make_hist(simulate_exp1(make_policies, prm, N))
            catch
                # println("Error, skipping")
                missing
            end
        end
    end;
end

function compute_loss(histograms, prms)
    tbl = DataFrame(prms)
    human = sum(human_hist; dims=:judgement)
    results = @showprogress "loss " pmap(histograms, prms) do hist, prm
        ismissing(hist) && return [Inf, 1000., 1000.]  # 1000 gives super long RT (a warning flag)
        model = sum(hist; dims=:judgement)
        if hasfield(typeof(prm), :α_ndt)
            (;α_ndt, θ_ndt) = prm
            lk = likelihood(model, human, Gamma(α_ndt, θ_ndt))
            [lk, α_ndt, θ_ndt]
        else
            res = optimize_ndt(model, human)
            [res.minimum; res.minimizer]
        end
    end
    tbl.loss, tbl.α_ndt, tbl.θ_ndt = invert(results)
    sort!(tbl, :loss)
end

# %% ==================== parameterization ====================

function reparameterize(prm)
    drift_σ = √(prm.between_σ^2 + prm.within_σ^2)
    prm = (;prm..., drift_σ)
    if hasfield(typeof(prm), :αθ_stop)
        prm = (;prm..., θ_stop = prm.αθ_stop / prm.α_stop)
    end
    if hasfield(typeof(prm), :αθ_ndt)
        prm = (;prm..., θ_ndt = prm.αθ_ndt / prm.α_ndt)
    end
    prm
end

sample_params(box, N) = map(reparameterize, sobol(N, box))

