@everywhere include("common.jl")
@everywhere include("exp1_simulate.jl")
mkpath("results/exp1")
# Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
# %% --------

pretest = load_data("exp1/pretest")
trials = load_data("exp1/trials")

@everywhere trials = $trials
@everywhere pretest = $pretest

N_SOBOL = 100_000

# %% ==================== define metrics ====================

@everywhere mean_or_missing(x; min_n=10) = length(x) < min_n ? missing : mean(x)
@everywhere function exp1_sumstats(trials)
    # skip_rate = @bywrap trials :pretest_accuracy mean(:response_type .== "empty")
    # pretest_rate = @bywrap trials :pretest_accuracy length(:rt) / nrow(trials)
    try
        p_correct = mapreduce(hcat, 0:200:10000) do cutoff
            @chain trials begin
                @bywrap :pretest_accuracy mean((:response_type .== "correct") .& (:rt .≤ cutoff))
            end
        end

        p_skip = mapreduce(hcat, 0:200:10000) do cutoff
            @chain trials begin
                @rsubset !(:response_type == "correct" && :rt ≤ cutoff)
                @bywrap :pretest_accuracy mean(:rt .≤ cutoff)
            end
        end

        (;
            p_correct, p_skip,
            acc_rt = @bywrap(trials, [:response_type, :pretest_accuracy], mean_or_missing(:rt)),
            acc_n = @bywrap(trials, [:response_type, :pretest_accuracy], length(:rt)),
            judge_rt = @bywrap(trials, [:response_type, :judgement], mean_or_missing(:rt)),
            judge_n = @bywrap(trials, [:response_type, :judgement], length(:rt)),
            rt_μ = @bywrap(trials, :response_type, mean(:rt)),
            rt_σ = @bywrap(trials, :response_type, std(:rt)),
        )
    catch
        missing
    end
end

target = exp1_sumstats(trials)

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

function cum_prob_loss(x)    
    mean(squared.(target.p_correct .- x.p_correct)) +
    mean(squared.(target.p_skip .- x.p_skip))
end

function acc_judge_loss(x)
    acc_rt_loss(x) + judge_rt_loss(x)
end

# %% ==================== optimal ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp1_mdp(prm)),
)

opt_prms = sobol(N_SOBOL, Box(
    drift_μ = (-1, 0.),
    noise = (1., 3.),
    drift_σ = (1, 3),
    threshold = (5, 10),
    sample_cost = (0, .05),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    judgement_noise=1,
))

# %% --------
opt_sumstats = @showprogress pmap(opt_prms) do prm
    exp1_sumstats(simulate_exp1(optimal_policies, prm))
end

serialize("tmp/exp1_opt_sumstats", opt_sumstats)
# %% --------
opt_ss = deserialize("tmp/exp1_opt_sumstats");
# %% --------
fit_prm, tbl = minimize_loss(cum_prob_loss, opt_ss, opt_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise]))[1:13, :])
df = simulate_exp1(optimal_policies, fit_prm)
@show cum_prob_loss(exp1_sumstats(df))
df |> CSV.write("results/exp1/optimal_trials.csv")

# %% ==================== empirical ====================

@everywhere begin
    const emp_pretest_stop_dist = empirical_distribution(@subset(pretest, :response_type .== "empty").rt)
    const emp_crit_stop_dist = empirical_distribution(@subset(trials, :response_type .== "empty").rt)

    empirical_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
        RandomStoppingPolicy(exp1_mdp(prm), emp_crit_stop_dist),
    )
end

emp_prms = sobol(N_SOBOL, Box(
    drift_μ = (-1, 1),
    noise = (1., 3.),
    drift_σ = (2., 5.),
    threshold = (1, 15),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    judgement_noise=1,
    sample_cost = 0.,
))

# %% --------
emp_ss = @showprogress pmap(emp_prms) do prm
    exp1_sumstats(simulate_exp1(empirical_policies, prm))
end
serialize("tmp/exp1_emp_sumstats", emp_ss)
# %% --------
emp_ss = deserialize("tmp/exp1_emp_sumstats")
# %% --------
emp_prm, tbl = minimize_loss(cum_prob_loss, emp_ss, emp_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise, :sample_cost]))[1:13, :])
df = simulate_exp1(empirical_policies, emp_prm)
@show cum_prob_loss(exp1_sumstats(df))
df |> CSV.write("results/exp1/empirical_trials.csv")

# %% ==================== random ====================

@everywhere function random_policies(prm)
    pretest_stop_dist = Gamma(prm.α_pre, prm.θ_pre)
    crit_stop_dist = Gamma(prm.α_crit, prm.θ_crit)
    (
        RandomStoppingPolicy(pretest_mdp(prm), pretest_stop_dist),
        RandomStoppingPolicy(exp1_mdp(prm), crit_stop_dist),
    )
end

rand_prms = sobol(N_SOBOL, Box(
    α_pre = (1, 20),
    θ_pre = (1, 20),
    α_crit = (1, 20),
    θ_crit = (1, 20),
    drift_μ = (-1, 1),
    noise = (1., 3.),
    drift_σ = (2., 5.),
    threshold = (1, 15),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    judgement_noise=1,
    sample_cost = 0.,
));

# %% --------
rand_ss = @showprogress pmap(rand_prms) do prm
    exp1_sumstats(simulate_exp1(random_policies, prm))
end;
serialize("tmp/exp1_rand_sumstats", rand_ss)
# %% --------
rand_ss = deserialize("tmp/exp1_rand_sumstats");
# %% --------
rand_prm, tbl = minimize_loss(cum_prob_loss, rand_ss, rand_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise, :sample_cost])))
df = simulate_exp1(random_policies, rand_prm)
@show cum_prob_loss(exp1_sumstats(df))
df |> CSV.write("results/exp1/random_trials.csv")
