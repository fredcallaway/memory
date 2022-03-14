if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end
# %% --------
@everywhere include("common.jl")
@everywhere include("exp1_base.jl")
mkpath("results/exp1")

N_SOBOL = 50_000
# %% --------
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

target = exp1_sumstats(trials);

function loss(ss)
    crossentropy(target, normalize(ss))
end

# %% ==================== optimal ====================

println("--- optimal ---")

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp1_mdp(prm)),
)

N_SOBOL = 50000
opt_prms = sobol(N_SOBOL, Box(
    drift_μ = (-0.5, 0.5),
    noise = (0.5, 2.),
    drift_σ = (1., 2.),
    threshold = (7, 15),
    sample_cost = (.005, .02),
    strength_drift_μ = 0,
    strength_drift_σ = (0, 1.),
    judgement_noise=1,
));

opt_sumstats = compute_sumstats("opt", optimal_policies, opt_prms);
# %% --------
acc_rt(ss) = @bywrap ss.rt [:pretest_accuracy, :response_type] mean(:μ, Weights(:n))
judge_rt(ss) = @bywrap ss.rt [:judgement, :response_type] mean(:μ, Weights(:n))
acc_judge_rt(ss) = @bywrap ss.rt [:pretest_accuracy, :judgement] mean(:μ, Weights(:n))

function loss(ss; ε=1e-3)
    mse(acc_rt(target), acc_rt(ss)) + mse(judge_rt(target), judge_rt(ss))    
end

opt_prm, opt_ss, tbl, full_loss = minimize_loss(loss, opt_sumstats, opt_prms) 
judge_rt(opt_ss)
judge_rt(target)
acc_rt(opt_ss)
acc_rt(target)

# %% --------

function loss(ss; ε=1e-3)
    crossentropy(target.hist, normalize(ss.hist .+ ε))
end
opt_prm, opt_ss, tbl, full_loss = minimize_loss(loss, opt_sumstats, opt_prms) 


display(select(tbl, Not([:strength_drift_μ, :judgement_noise]))[1:13, :])
df = simulate_exp1(optimal_policies, opt_prm)
@show loss(exp1_sumstats(df))
CSV.write("results/exp1/optimal_trials.csv", df)

# %% --------

figure() do
    plot(ssum(target(response_type="empty", pretest_accuracy=0.5), :judgement))
end

figure() do
    plot(ssum(target("empty"), :judgement))
end

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
    drift_μ = (-0.5, 1.0),
    noise = (2., 4.),
    drift_σ = (0.5, 1.5),
    threshold = (7, 15),
    sample_cost = (.01, .05),
    strength_drift_μ = 0,
    strength_drift_σ = (0, 1),
    judgement_noise=1,
));


emp_sumstats = compute_sumstats("emp", empirical_policies, emp_prms);

# %% --------

emp_prm, emp_ss, tbl, full_loss = minimize_loss(loss, emp_sumstats, emp_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise, :sample_cost]))[1:13, :])
df = simulate_exp1(empirical_policies, emp_prm)
@show loss(exp1_sumstats(df))
CSV.write("results/exp1/empirical_trials.csv", df)

# %% ==================== decision bound ====================

println("--- decision bound ---")

@everywhere bound_policies(prm) = (
    ConstantBoundPolicy(pretest_mdp(prm), prm.θ),
    ConstantBoundPolicy(exp1_mdp(prm), prm.θ),
)

bound_prms = sobol(N_SOBOL, Box(
    drift_μ = (-1, 1),
    noise = (1., 3.),
    drift_σ = (0.5, 2.5),
    threshold = (5, 15),
    θ = (1, 15),
    τ = (.001, 1, :log),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    judgement_noise=1,
    sample_cost = 0.,
))


bound_sumstats = compute_sumstats("bound", bound_policies, bound_prms);

# %% --------

bound_prm, bound_ss, tbl, full_loss = minimize_loss(loss, bound_sumstats, bound_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise, :sample_cost]))[1:13, :])
df = simulate_exp1(bound_policies, bound_prm)
@show loss(exp1_sumstats(df))
CSV.write("results/exp1/bound_trials.csv", df)



