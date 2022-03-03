@everywhere include("common.jl")
@everywhere include("exp1_simulate.jl")
mkpath("results/exp1")

if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end
# %% --------

pretest = load_data("exp1/pretest")
trials = load_data("exp1/trials")

@everywhere trials = $trials
@everywhere pretest = $pretest

N_SOBOL = 100_000
include("exp1_loss.jl")

function compute_sumstats(name, make_policies, prms)
    mkpath("cache/exp1_$(name)_sumstats")
    @showprogress pmap(prms) do prm
        cache("cache/exp1_$(name)_sumstats/$(stringify(prm))") do
            exp1_sumstats(simulate_exp1(make_policies, prm))
        end
    end;
end

function loss(ss)
    mae(target.unrolled(time = <(10000)), ss.unrolled(time = <(10000)))
end

# %% ==================== optimal ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp1_mdp(prm)),
)

opt_prms = sobol(N_SOBOL, Box(
    drift_μ = (-1, 1.),
    noise = (1., 3.),
    drift_σ = (1, 3),
    threshold = (5, 15),
    sample_cost = (0, .05),
    ndt_α = (1, 100, :log),
    ndt_μ = (0, 200),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    judgement_noise=1,
))

opt_sumstats = compute_sumstats("opt", optimal_policies, opt_prms)

# %% --------

opt_prm, opt_ss, tbl, full_loss = minimize_loss(loss, opt_sumstats, opt_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise]))[1:13, :])
df = simulate_exp1(optimal_policies, opt_prm)
@show loss(exp1_sumstats(df))
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

emp_sumstats = compute_sumstats("emp", empirical_policies, emp_prms)

# %% --------

emp_prm, emp_ss, tbl, full_loss = minimize_loss(loss, emp_sumstats, emp_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise, :sample_cost]))[1:13, :])
df = simulate_exp1(empirical_policies, emp_prm)
@show loss(exp1_sumstats(df))
df |> CSV.write("results/exp1/empirical_trials.csv")

# # %% ==================== random ====================

# @everywhere function random_policies(prm)
#     pretest_stop_dist = Gamma(prm.α_pre, prm.θ_pre)
#     crit_stop_dist = Gamma(prm.α_crit, prm.θ_crit)
#     (
#         RandomStoppingPolicy(pretest_mdp(prm), pretest_stop_dist),
#         RandomStoppingPolicy(exp1_mdp(prm), crit_stop_dist),
#     )
# end

# rand_prms = sobol(N_SOBOL, Box(
#     α_pre = (1, 20),
#     θ_pre = (1, 20),
#     α_crit = (1, 20),
#     θ_crit = (1, 20),
#     drift_μ = (-1, 1),
#     noise = (1., 3.),
#     drift_σ = (2., 5.),
#     threshold = (1, 15),
#     strength_drift_μ = 0,
#     strength_drift_σ = 0.,
#     judgement_noise=1,
#     sample_cost = 0.,
# ));

# # %% --------
# rand_ss = @showprogress pmap(rand_prms) do prm
#     exp1_sumstats(simulate_exp1(random_policies, prm))
# end;
# serialize("tmp/exp1_rand_sumstats", rand_ss)
# # %% --------
# rand_ss = deserialize("tmp/exp1_rand_sumstats");
# # %% --------
# rand_prm, tbl = minimize_loss(cum_prob_loss, rand_ss, rand_prms);
# display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise, :sample_cost])))
# df = simulate_exp1(random_policies, rand_prm)
# @show cum_prob_loss(exp1_sumstats(df))
# df |> CSV.write("results/exp1/random_trials.csv")
