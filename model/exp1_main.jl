@everywhere include("common.jl")
@everywhere include("exp1_simulate.jl")
mkpath("results/exp1")
Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
# %% --------

pretest = load_data("exp1/pretest")
trials = load_data("exp1/trials")

@everywhere trials = $trials

# %% ==================== Metrics ====================

@everywhere mean_or_missing(x; min_n=10) = length(x) < min_n ? missing : mean(x)
@everywhere function exp1_metrics(df)
    skip_rate = @bywrap df :pretest_accuracy mean(:response_type .== "empty")
    pretest_rate = @bywrap df :pretest_accuracy length(:rt) / nrow(df)
    full_rt = @bywrap df [:response_type, :pretest_accuracy] mean_or_missing(:rt)
    full_n = @bywrap df [:response_type, :pretest_accuracy] length(:rt)
    rt_μ = @bywrap df :response_type mean(:rt)
    rt_σ = @bywrap df :response_type std(:rt)
    (;skip_rate, pretest_rate, rt_μ, rt_σ, full_rt, full_n)
end

target = exp1_metrics(trials)

# %% ==================== Arbitrary ====================

prm = (
    drift_μ = 0.,
    drift_σ = 1.5,
    strength_drift_μ = 0,
    strength_drift_σ = 0,
    noise = 1.5,
    threshold = 10,
    sample_cost = .04
)

# prm = (threshold = 2.5, drift_σ = 1.1666666666666665, drift_μ = 0.0, sample_cost = 0.01,  noise = 0.6,
#  strength_drift_σ = 0.0, strength_drift_μ = 0)

df = simulate_exp1(prm)

figure() do
    plot(
        plot(exp1_metrics(df).full_rt', ylim=(0, 3000)),
        plot(exp1_metrics(trials).full_rt', ylim=(0, 3000)),
        legend=false
    )
end

df |> CSV.write("results/exp1/optimal_trials.csv")

# %% ==================== Fitted ====================

prms = sobol(10000, Box(
    drift_μ = (-1, 1),
    drift_σ = (1, 3),
    threshold = (1, 10),
    sample_cost = (0, .02),
    noise = (.2, 2),
    strength_drift_μ = 0,
    strength_drift_σ = (0, 0.5)
))

metrics = @showprogress pmap(prms) do prm
    exp1_metrics(simulate_exp1(prm))
    # cache(".cache/exp1_metrics/$(hash(prm))") do
    # end
end

# %% --------
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

serialize("tmp/metrics", metrics)
# %% --------

L = map(metrics) do pred
    size(pred.full_rt) == size(target.full_rt) || return Inf
    dif = (target.full_rt .- pred.full_rt)[:]
    l = sum(squared.(dif[(!ismissing).(target.full_rt[:])]) / 1000^2)
    ismissing(l) && return Inf
    L + mean(squared.(target.skip_rate .- pred.skip_rate))
end;

# %% --------
# WAS A BUG HERE!!! NEED TO RETRY OLD FITTING
flat_prms = collect(prms)[:];
flat_L = collect(L)[:];
res = flat_prms[partialsortperm(flat_L, 1:10)] |> DataFrame
res.loss = partialsort(flat_L, 1:10)
res

fit_prm = flat_prms[argmin(flat_L)]

df = simulate_exp1(fit_prm)
pred = exp1_metrics(df)
df |> CSV.write("results/exp1/optimal_trials.csv")
