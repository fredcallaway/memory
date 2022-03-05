if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

@everywhere include("common.jl")
@everywhere include("exp1_base.jl")
mkpath("results/exp1")

N_SOBOL = 10_000

function compute_sumstats(name, make_policies, prms; read_only = false)
    mkpath("cache/exp1_$(name)_sumstats")
    map = read_only ? asyncmap : pmap
    @showprogress map(prms) do prm
        cache("cache/exp1_$(name)_sumstats/$(hash(prm))"; read_only) do
            exp1_sumstats(simulate_exp1(make_policies, prm))
        end
    end;
end

# %% ==================== load data ====================

pretest = load_data("exp1/pretest")
trials = load_data("exp1/trials")

@everywhere trials = $trials
@everywhere pretest = $pretest

target = exp1_sumstats(trials)

function loss(ss)
    mae(target.unrolled, ss.unrolled)
end

# %% ==================== optimal ====================
println("--- optimal ---")

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp1_mdp(prm)),
)

opt_prms = sobol(N_SOBOL, Box(
    drift_μ = (-2, 0.),
    noise = (1., 5.),
    drift_σ = (1, 3),
    threshold = (3, 10),
    sample_cost = (0, .03),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    judgement_noise=1,
))

opt_sumstats = compute_sumstats("opt", optimal_policies, opt_prms)

opt_prm, opt_ss, tbl, full_loss = minimize_loss(loss, opt_sumstats, opt_prms);

display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise]))[1:13, :])

df = simulate_exp1(optimal_policies, opt_prm)
@show loss(exp1_sumstats(df))
CSV.write("results/exp1/optimal_trials.csv", df)

# %% --------

pretest_rate(ss) = @chain ss.rt begin
     @bywrap :pretest_accuracy sum(:n)
     normalize
end

acc_rate(ss) = @chain ss.rt begin
     @bywrap [:pretest_accuracy, :response_type] sum(:n)
     normalize
end

acc_rt(ss) = @bywrap ss.rt [:pretest_accuracy, :response_type] (sum(:n) > 10 ? mean(:μ, Weights(:n)) : missing)

# %% --------
opt_prm, opt_ss, tbl, full_loss = minimize_loss(opt_sumstats, opt_prms) do ss
    mae(acc_rate(ss), acc_rate(target)) <= .05 || return Inf
    x = acc_rt(target)[:]
    mask = (!ismissing).(x)
    mae(x[mask], acc_rt(ss)[mask])
end

acc_rt(opt_ss)
acc_rate(opt_ss)
acc_rate(target)

# %% --------

df = simulate_exp1(optimal_policies, opt_prm)
CSV.write("results/exp1/optimal_trials.csv", df)

# %% ==================== empirical ====================
println("--- empirical ---")

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

emp_sumstats = compute_sumstats("emp", empirical_policies, emp_prms)

# %% --------

emp_prm, emp_ss, tbl, full_loss = minimize_loss(loss, emp_sumstats, emp_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise, :sample_cost]))[1:13, :])
df = simulate_exp1(empirical_policies, emp_prm)
@show loss(exp1_sumstats(df))
CSV.write("results/exp1/empirical_trials.csv", df)
