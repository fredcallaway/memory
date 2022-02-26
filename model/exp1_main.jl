@everywhere include("common.jl")
@everywhere include("exp1_simulate.jl")
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
    sum(squared.(pre_target.acc_rate .- pred.acc_rate))
    # squared((pre_target.rt_μ - pred.rt_μ) / 3pre_target.rt_σ) +
    # squared((pre_target.rt_σ - pred.rt_σ) / 3pre_target.rt_σ)
end;

flat_prms = collect(pre_prms)[:];
flat_L = collect(L)[:];
flat_prms[sortperm(flat_L)]
pre_prm = keymin(L)

pre_metrics[argmin(L)]
pre_target

# account for the change of adding implicit cost
# re_prm = (;pre_prm..., sample_cost = 0.007071067811865478 - 0.00333333) 


# %% ==================== Fit critical ====================

@everywhere function exp1_metrics(df)
    skip_rate = @bywrap df :pretest_accuracy mean(:response_type .== "empty")
    pretest_rate = @bywrap df :pretest_accuracy length(:rt) / nrow(df)
    rt_μ = @bywrap df :response_type mean(:rt)
    rt_σ = @bywrap df :response_type std(:rt)
    (;skip_rate, pretest_rate, rt_μ, rt_σ)
end



@everywhere function exp1_metrics(df)
    skip_rate = @bywrap df :pretest_accuracy mean(:response_type .== "empty")
    pretest_rate = @bywrap df :pretest_accuracy length(:rt) / nrow(df)
    rt_μ = @bywrap df :response_type mean(:rt)
    rt_σ = @bywrap df :response_type std(:rt)
    (;skip_rate, pretest_rate, rt_μ, rt_σ)
end

# %% --------

prms = grid(7, Box(
    drift_μ = (-1, 1),
    drift_σ = (0.5, 1.5),
    threshold = (1, 10),
    sample_cost = (0, .01),
    noise = (.2, 2),
    strength_drift_μ = 0,
    strength_drift_σ = 1e-9

))

metrics = @showprogress pmap(prms) do prm
    cache(".cache/exp1_metrics/$(stringify(prm))") do
        exp1_metrics(simulate_exp1(prm))
    end
end

# %% --------
target = exp1_metrics(trials)

L = map(metrics) do pred
    try
        mean(squared.(target.skip_rate .- pred.skip_rate)) +
        mean(squared.(target.pretest_rate .- pred.pretest_rate)) +
        mean(squared.((target.rt_μ .- pred.rt_μ) ./ target.rt_σ)) +
        mean(squared.((target.rt_σ .- pred.rt_σ) ./ target.rt_σ))
    catch
        Inf
    end
end;

flat_prms = collect(pre_prms)[:];
flat_L = collect(L)[:];
res = flat_prms[partialsortperm(flat_L, 1:10)] |> DataFrame
res.loss = partialsort(flat_L, 1:10)
fit_prm = keymin(L)


exp1_metrics(simulate_exp1(fit_prm))
target
simulate_exp1(fit_prm) |> CSV.write("results/exp1/optimal_trials.csv")

# %% ==================== Use pretest parameters ====================

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
