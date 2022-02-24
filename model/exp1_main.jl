@everywhere include("common.jl")
mkpath("results/exp1")
Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
# %% --------

pretest = load_data("exp1/pretest")
trials = load_data("exp1/trials")

@everywhere trials = $trials

# %% ==================== Fit pretest ====================

pre_prms = grid(5, Box(
    drift_μ = (-1, 1),
    drift_σ = (0.5, 1.5),
    threshold = (1, 10),
    sample_cost = (.001, .05, :log),
    noise = (.2, 2),
))

mkpath(".cache/exp1_pre_metrics")
pre_metrics = @showprogress pmap(pre_prms) do prm
    cache(".cache/exp1_pre_metrics/$(stringify(prm))") do
        pretest_metrics(simulate_pretest(prm))
    end
end

pre_target = pretest_metrics(pretest)

# %% --------
L = map(pre_metrics) do pred
    sum(squared.(pre_target.acc_rate .- pred.acc_rate)) +
    squared((pre_target.rt_μ - pred.rt_μ) / 3pre_target.rt_σ) +
    squared((pre_target.rt_σ - pred.rt_σ) / 3pre_target.rt_σ)
end;

flat_prms = collect(pre_prms)[:];
flat_L = collect(L)[:];
flat_prms[sortperm(flat_L)]
pre_prm = keymin(L)

pre_metrics[argmin(L)]
pre_target

# %% ==================== Simulate data ====================

@everywhere function discretize_judgement!(df)
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

@everywhere function simulate_exp1(pre_pol, crit_pol, N=10000; strength_drift=Normal(0, 1e-9))
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

# %% ==================== Fit critical ====================

function pretest_mdp(prm)
    MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=1,
        prm.threshold, prm.sample_cost, prm.noise,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end

function exp1_mdp(prm)
    MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=3,
        prm.threshold, prm.sample_cost, prm.noise,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end

# .1 cents per second
# each time step if 200ms
explicit_sample_cost = (ms_per_sample / 1000) * .1
prm = (;pre_prm..., sample_cost=pre_prm.sample_cost + explicit_sample_cost)

pre_pol = OptimalPolicy(pretest_mdp(pre_prm))
crit_pol = OptimalPolicy(exp1_mdp(prm))
df = simulate_exp1(pre_pol, crit_pol)
df |> CSV.write("results/exp1/optimal_trials.csv")

# %% ==================== Fit parameters ====================

@everywhere function compute_metrics(df)
    (q_rt = quantile(df.rt, .1:.2:.9),
     q_rt_correct = quantile(@subset(df, :response_type .== "correct").rt, .1:.2:.9),
     q_rt_skip = quantile(@subset(df, :response_type .== "empty").rt, .1:.2:.9),
     p_correct = mean(df.response_type .== "correct"))
end

@everywhere function simulate_optimal(prm::NamedTuple, N=10000)
    m = MetaMDP{1}(;allow_stop=true, miss_cost=3, max_step=60, prm.threshold,
                   prior=Normal(prm.drift_μ, prm.drift_σ), prm.sample_cost, noise=prm.noise*prm.drift_σ)
    
    OptimalPolicy(m; dv=prm.threshold*.01) |> make_frame
end

# @fetchfrom 2 simulate_optimal((drift_μ=0, drift_σ=1, sample_cost=.06, noise=1.5, threshold=7)) |> compute_metrics
# simulate_optimal((drift_μ=0, drift_σ=1/7, sample_cost=.06, noise=1.5/7, threshold=1)) |> compute_metrics

# simulate_optimal((drift_μ=0, drift_σ=2, sample_cost=.06, noise=2, threshold=14)) |> compute_metrics
# simulate_optimal((drift_μ=0, drift_σ=.5, sample_cost=.06, noise=.5, threshold=3.5)) |> compute_metrics

# %% --------

prms = grid(
    drift_μ=[0.],
    drift_σ=.5:.1:1.5,
    sample_cost=.01:.05,
    noise=.2:1.2,
    threshold=3:10
)

@everywhere target = compute_metrics(trials)

metrics = @showprogress pmap(prms) do prm
    compute_metrics(simulate_optimal(prm))
end;

squared(x) = x^2
L = map(metrics) do m
    # sum(squared.(m.q_rt_correct .- target.q_rt_correct)) +
    # sum(squared.(m.q_rt_correct .- target.q_rt_correct)) +
    sum(abs.(m.q_rt .- target.q_rt)) +
    abs((m.p_correct - target.p_correct)) * 100_000^2
end;

# %% --------
prms[argmin(L)]
metrics[argmin(L)]

simulate_optimal(prms[argmin(L)]) |> CSV.write("results/exp1/optimal_trials.csv")

# %% ==================== Hand-chosen ====================


m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.06, 
    threshold=7, noise=1.5, max_step=60, prior=Normal(0, 1)
)

opt_pol = OptimalPolicy(m)
df = make_frame(opt_pol)
df |> CSV.write("results/exp1/optimal_trials.csv")

rt = @subset(trials, :response_type .== "empty").rt
rand_pol = StopDistributionPolicy2(m, fit(Gamma, rt ./ ms_per_sample))
make_frame(rand_pol) |> CSV.write("results/exp1/random_trials.csv")
