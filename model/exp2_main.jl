if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

@everywhere include("common.jl")
@everywhere include("exp2_base.jl")
mkpath("results/exp2")

# %% ==================== load data ====================

pretest = CSV.read("../data/processed/exp2/pretest.csv", DataFrame, missingstring="NA")
trials = CSV.read("../data/processed/exp2/trials.csv", DataFrame, missingstring="NA")
fixations = CSV.read("../data/processed/exp2/fixations.csv", DataFrame, missingstring="NA")

pretest = @rsubset pretest :practice == false :block == 3
trials = @chain trials begin
    @rsubset :n_pres > 0
    @rsubset :response_type != "intrusion"
    @rtransform :choose_first = :response_type == "correct" ? :choose_first : missing
end

@everywhere trials = $trials
@everywhere pretest = $pretest
@everywhere fixations = $fixations

# %% ==================== optimal ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp2_mdp(prm)),
)

opt_prm = deserialize("tmp/exp1_opt_prm")
opt_prm = (;opt_prm..., switch_cost=.01)
@time opt_df = simulate_exp2(optimal_policies, prm, 1000000);
opt_dur_noise = mle_duration_noise(opt_df, fixations)
add_duration_noise!(opt_df, opt_dur_noise)

opt_trials = make_trials(opt_df); opt_fixations = make_fixations(opt_df)
CSV.write("results/exp2/optimal_trials.csv", opt_trials)
CSV.write("results/exp2/optimal_fixations.csv", opt_fixations)

# %% ==================== empirical ====================

plausible_skips(x) = @rsubset(x, :response_type in ["other", "empty"])
const emp_pretest_stop_dist = empirical_distribution(plausible_skips(pretest).rt)
const emp_crit_stop_dist = empirical_distribution(skipmissing(plausible_skips(trials).rt))
const emp_switch_dist = empirical_distribution(fixations.duration)

empirical_policies(prm) = (
    RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
    RandomSwitchingPolicy(exp2_mdp(prm), emp_switch_dist, emp_crit_stop_dist),
)

emp_prm = deserialize("tmp/exp1_emp_emp_prm")
emp_prm = (;emp_prm..., switch_cost=NaN)
@time emp_df = simulate_exp2(empirical_policies, emp_prm, 1000000);
emp_dur_noise = mle_duration_noise(emp_df, fixations)
add_duration_noise!(emp_df, emp_dur_noise)

emp_trials = make_trials(emp_df); emp_fixations = make_fixations(emp_df)
CSV.write("results/exp2/empirical_trials.csv", emp_trials)
CSV.write("results/exp2/empirical_fixations.csv", emp_fixations)
