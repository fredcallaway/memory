@everywhere include("common.jl")
@everywhere include("exp1_simulate.jl")
mkpath("results/exp1")
Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
# %% --------

pretest = load_data("exp1/pretest")
trials = load_data("exp1/trials")

@everywhere trials = $trials
@everywhere pretest = $pretest

# %% ==================== Metrics ====================

@everywhere mean_or_missing(x; min_n=10) = length(x) < min_n ? missing : mean(x)
@everywhere function exp1_metrics(df)
    # skip_rate = @bywrap df :pretest_accuracy mean(:response_type .== "empty")
    # pretest_rate = @bywrap df :pretest_accuracy length(:rt) / nrow(df)
    (
        acc_rt = @bywrap(df, [:response_type, :pretest_accuracy], mean_or_missing(:rt)),
        acc_n = @bywrap(df, [:response_type, :pretest_accuracy], length(:rt)),
        judge_rt = @bywrap(df, [:response_type, :judgement], mean_or_missing(:rt)),
        judge_n = @bywrap(df, [:response_type, :judgement], length(:rt)),
        rt_μ = @bywrap(df, :response_type, mean(:rt)),
        rt_σ = @bywrap(df, :response_type, std(:rt)),
    )
end

target = exp1_metrics(trials)

# %% ==================== Manual ====================

function cost_to_wage(sample_cost)
    cps = .01* sample_cost / (ms_per_sample/1000)
    dps = cps / 100
    sph = 60 * 60
    dph = cps * sph
end

prm = (
    drift_μ = 0.,
    drift_σ = 1.5,
    strength_drift_μ = 0,
    strength_drift_σ = 0,
    noise = 1.5,
    threshold = 10,
    sample_cost = .05,
    judgement_noise = 1,
)

df = simulate_exp1(prm)

figure() do
    plot(
        plot(exp1_metrics(df).acc_rt'),
        plot(target.acc_rt'),
        plot(exp1_metrics(df).judge_rt'),
        plot(target.judge_rt'),
        legend=false,
        size=(600, 600),
        ylim=(0, 4000),
    )
end

# %% --------

df |> CSV.write("results/exp1/optimal_trials.csv")

# %% ==================== Automatic ====================

prms = sobol(10000, Box(
    drift_μ = (-1, 1),
    noise = (.5, 2.5),
    drift_σ = (1, 3),
    threshold = (5, 15),
    sample_cost = (0, .1),
    strength_drift_μ = 0,
    # strength_drift_σ = (0, 0.5),
    strength_drift_σ = 0.,
    judgement_noise=1,
))

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp1_mdp(prm)),
)

# %% --------
metrics = @showprogress pmap(prms) do prm
    exp1_metrics(simulate_exp1(optimal_policies, prm))
end
serialize("tmp/exp1_opt_metrics_nodrift", metrics)
# %% --------
metrics = deserialize("tmp/exp1_opt_metrics_nodrift");

# %% --------

response_rate(x) = x.acc_n(response_type="correct") ./ ssum(x.acc_n, :response_type)
pretest_dist(x) = normalize(ssum(x.acc_n, :response_type))

function marginal_loss(pred)
    size(pred.acc_rt) == size(target.acc_rt) || return Inf
    sum(squared.(response_rate(pred) .- response_rate(target))) +
    sum(squared.(pretest_dist(pred) .- pretest_dist(target))) +
    mean(squared.((target.rt_μ .- pred.rt_μ) ./ target.rt_σ))
end

function acc_rt_loss(pred)
    size(pred.acc_rt) == size(target.acc_rt) || return Inf
    dif = (target.acc_rt .- pred.acc_rt)[:]
    mask = (!ismissing).(target.acc_rt[:])
    l = mean(squared.(dif[mask]) / 1000^2)
    ismissing(l) ? Inf : l
end

function judge_rt_loss(pred)
    size(pred.judge_rt) == size(target.judge_rt) || return Inf
    dif = (target.judge_rt .- pred.judge_rt)[:]
    mask = (!ismissing).(target.judge_rt[:])
    l = mean(squared.(dif[mask]) / 1000^2)
    ismissing(l) ? Inf : l
end

function minimize_loss(loss, metrics, prms)
    L = map(loss, metrics);
    flat_prms = collect(prms)[:];
    flat_L = collect(L)[:];
    fit_prm = flat_prms[argmin(flat_L)]
    tbl = flat_prms[partialsortperm(flat_L, 1:10)] |> DataFrame
    tbl.loss = partialsort(flat_L, 1:10)
    fit_prm, tbl
end


# TODO: fix
# fit_prm, tbl = minimize_loss(acc_rt_loss, metrics, prms);
fit_prm, tbl = minimize_loss(x->acc_rt_loss(x) + judge_rt_loss(x), metrics, prms);
println(tbl)
df = simulate_exp1(optimal_policies, fit_prm)
pred = exp1_metrics(df)
df |> CSV.write("results/exp1/optimal_trials.csv")
serialize("tmp/exp1_fit_prm2", fit_prm)

fit_prm
# %% ==================== Random ====================

@everywhere begin
    pretest_stop_dist = fit(Gamma, @subset(pretest, :response_type .== "empty").rt ./ ms_per_sample)
    crit_stop_dist = fit(Gamma, @subset(trials, :response_type .== "empty").rt ./ ms_per_sample)
    random_policies(prm) = (
        StopDistributionPolicy2(pretest_mdp(prm), pretest_stop_dist),
        StopDistributionPolicy2(exp1_mdp(prm), crit_stop_dist),
    )
end

prms = sobol(10000, Box(
    drift_μ = (-1, 1),
    noise = (.5, 2.5),
    drift_σ = (1, 3),
    threshold = (5, 15),
    strength_drift_μ = 0,
    strength_drift_σ = (0, 0.5),
    judgement_noise=1,
    sample_cost = 0.,
))

metrics = @showprogress pmap(prms) do prm
    exp1_metrics(simulate_exp1(random_policies, prm))
end
serialize("tmp/metrics", metrics)
metrics = deserialize("tmp/metrics")
# %% --------
fit_prm, tbl = minimize_loss(acc_rt_loss, metrics, prms);
tbl

# make_frame(rand_pol) |> CSV.write("results/exp1/random_trials.csv")
simulate_exp1(random_policies, prm) |> CSV.write("results/exp1/random_trials.csv")




