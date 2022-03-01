@everywhere include("common.jl")
@everywhere include("exp1_simulate.jl")
mkpath("results/exp1")
Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
# %% --------

pretest = load_data("exp1/pretest")
trials = load_data("exp1/trials")

@everywhere trials = $trials
@everywhere pretest = $pretest

# %% ==================== define metrics ====================

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

# %% ==================== loss function ====================

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

function full_loss(x)    
    acc_rt_loss(x) + judge_rt_loss(x)
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

# %% ==================== optimal ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp1_mdp(prm)),
)

prms = sobol(10000, Box(
    drift_μ = (-1, 1),
    noise = (.5, 2.5),
    drift_σ = (1, 3),
    threshold = (5, 15),
    sample_cost = (0, .1),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    judgement_noise=1,
))

# %% --------
metrics = @showprogress pmap(prms) do prm
    exp1_metrics(simulate_exp1(optimal_policies, prm))
end
serialize("tmp/exp1_opt_metrics", metrics)
# %% --------
metrics = deserialize("tmp/exp1_opt_metrics");
# %% --------
fit_prm, tbl = minimize_loss(full_loss, metrics, prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise])))
df = simulate_exp1(optimal_policies, fit_prm)
full_loss(exp1_metrics(df))
df |> CSV.write("results/exp1/optimal_trials.csv")

# %% ==================== random ====================

@everywhere function random_policies(prm)
    pretest_stop_dist = Gamma(prm.α_pre, prm.θ_pre)
    crit_stop_dist = Gamma(prm.α_crit, prm.θ_crit)
    (
        RandomStoppingPolicy(pretest_mdp(prm), pretest_stop_dist),
        RandomStoppingPolicy(exp1_mdp(prm), crit_stop_dist),
    )
end

rand_prms = sobol(10000, Box(
    α_pre = (1, 20),
    θ_pre = (1, 20),
    α_crit = (1, 20),
    θ_crit = (1, 20),
    drift_μ = (-1, 1),
    noise = (.5, 2.5),
    drift_σ = (1, 3),
    threshold = (5, 15),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    judgement_noise=1,
    sample_cost = 0.,
));

# %% --------
rand_metrics = @showprogress pmap(rand_prms) do prm
    exp1_metrics(simulate_exp1(random_policies, prm))
end;
serialize("tmp/exp1_rand_metrics", rand_metrics)
# %% --------
rand_metrics = deserialize("tmp/exp1_rand_metrics");
# %% --------
rand_prm, tbl = minimize_loss(full_loss, rand_metrics, rand_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise, :sample_cost])))
df = simulate_exp1(random_policies, rand_prm)
full_loss(exp1_metrics(df))
df |> CSV.write("results/exp1/random_trials.csv")

# %% ==================== empirical ====================

@everywhere begin
    const emp_pretest_stop_dist = empirical_distribution(@subset(pretest, :response_type .== "empty").rt)
    const emp_crit_stop_dist = empirical_distribution(@subset(trials, :response_type .== "empty").rt)

    empirical_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
        RandomStoppingPolicy(exp1_mdp(prm), emp_crit_stop_dist),
    )
end

emp_prms = sobol(10000, Box(
    drift_μ = (-1, 1),
    noise = (.5, 2.5),
    drift_σ = (1, 3),
    threshold = (5, 15),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    judgement_noise=1,
    sample_cost = 0.,
))

# %% --------
emp_metrics = @showprogress pmap(emp_prms) do prm
    exp1_metrics(simulate_exp1(empirical_policies, prm))
end
serialize("tmp/exp1_emp_metrics", emp_metrics)
# %% --------
emp_metrics = deserialize("tmp/exp1_emp_metrics")
# %% --------
emp_prm, tbl = minimize_loss(full_loss, emp_metrics, emp_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise, :sample_cost])))
df = simulate_exp1(empirical_policies, emp_prm)
exp1_metrics(df).judge_rt
df |> CSV.write("results/exp1/empirical_trials.csv")
