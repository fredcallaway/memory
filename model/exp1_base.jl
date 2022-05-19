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

# %% ==================== summary statistics ====================

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

function aggregate_rt(trials)
    @chain trials begin
        groupby([:response_type, :pretest_accuracy, :judgement])
        @combine begin
            :μ = mean(:rt)
            :σ = std(:rt)
            :n = length(:rt)
        end
    end
end

function exp1_sumstats(trials)
    (hist = make_hist(trials), rt = aggregate_rt(trials))
end

function compute_sumstats(name, make_policies, prms; N=100000, read_only = false)
    dir = "cache/$(RUN)_exp1_$(name)_sumstats_$N"
    mkpath(dir)
    map = read_only ? asyncmap : pmap
    @showprogress "sumstats" map(prms) do prm
        cache("$dir/$(hash(prm))"; read_only) do
            try
                exp1_sumstats(simulate_exp1(make_policies, prm, N))
            catch
                # println("Error, skipping")
                missing
            end
        end
    end;
end

# %% ==================== likelihood ====================

function smooth_rt!(result, p::KeyedArray, d::Distribution, ε::Float64=1e-6)
    pd = diff([0; cdf(d, p.rt)])
    for h in axes(p, 4), i in axes(p, 3), j in axes(p, 2), z in axes(p, 1)
        result[z, j, i, h] = sum(1:z) do k
            y = z - k
            @inbounds p[k, j, i, h] * pd[y + 1]
        end
    end
    smooth_uniform!(result, ε)
end

function optimize_rt_noise(ss)
    human = sum(target.hist; dims=:judgement)
    model = sum(ss.hist; dims=:judgement)
    X = zeros(size(model))
    optimize([10., 10.]) do x
        any(xi < 0 for xi in x) && return Inf
        smooth_rt!(X, model, Gamma(x...))
        crossentropy(human, X)
    end
end

function acc_rate(ss)
    x = ssum(ss.hist, :rt, :judgement)
    x ./= sum(x, dims=:response_type)
    x("correct")
end

function loss(ss)
    ismissing(ss) && return Inf
    x = acc_rate(ss)
    (x[1] < .1 && x[3] > .85) || return Inf
    res = optimize_rt_noise(ss)
    res.minimum
end

# %% ==================== parameterization ====================

function reparameterize(prm)
    drift_σ = √(prm.between_σ^2 + prm.within_σ^2)
    (;prm..., drift_σ)
end

sample_params(box) = map(reparameterize, sobol(N_SOBOL, box))


