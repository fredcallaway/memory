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

# %% ==================== duration smoothing ====================

function nonfinal_duration_hist(fixations; dt=ms_per_sample, maxt=10000)
    durations = @chain fixations begin
        @rsubset :presentation ≠ :n_pres
        @with :duration
    end
    p = initialize_keyed(0., duration=dt:dt:maxt)
    for d in durations
        i = min(Int(cld(d, dt)), length(p))
        p[i] += 1
    end
    p ./= sum(p)
end

function smooth_duration!(result, p::KeyedArray, d::Distribution; ε=1e-6)
    pd = diff([0; cdf(d, p.duration)])
    for z in axes(p, 1)
        result[z] = sum(1:z) do k
            y = z - k
            @inbounds p[k] * pd[y + 1]
        end
    end
    result .*= (1 - ε * length(result))
    result .+= ε
    result
end

function optimize_duration_noise(target, model)
    X = zeros(length(model))
    optimize([10., 10.]) do x
        any(xi < 0 for xi in x) && return Inf
        smooth_duration!(X, model, Gamma(x...))
        crossentropy(target, X)
    end
end

function add_duration_noise!(df)
    target = nonfinal_duration_hist(fixations)
    model = nonfinal_duration_hist(make_fixations(df))
    duration_noise = Gamma(optimize_duration_noise(target, model).minimizer...)
    @show duration_noise

    for x in df.presentation_times
        x .+= rand(duration_noise, length(x))
    end
end

# %% ==================== optimal ====================
@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp2_mdp(prm)),
)

prm = deserialize("tmp/exp1_opt_prm")
prm = (;prm..., switch_cost=.01)
@time df = simulate_exp2(optimal_policies, prm, 1000000);

opt_trials = make_trials(df); opt_fixations = make_fixations(df)
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

prm = deserialize("tmp/exp1_emp_prm")
prm = (;prm..., switch_cost=NaN)
@time df = simulate_exp2(empirical_policies, prm, 1000000);

emp_trials = make_trials(df); emp_fixations = make_fixations(df)
CSV.write("results/exp2/empirical_trials.csv", emp_trials)
CSV.write("results/exp2/empirical_fixations.csv", emp_fixations)


# # %% ==================== empirical ====================
# println("--- empirical ---")

# @everywhere begin
#     plausible_skips(x) = @rsubset(x, :response_type in ["other", "empty"])
#     const emp_pretest_stop_dist = empirical_distribution(plausible_skips(pretest).rt)
#     const emp_crit_stop_dist = empirical_distribution(plausible_skips(trials).rt)
#     const emp_switch_dist = empirical_distribution(fixations.duration)

#     empirical_policies(prm) = (
#         RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
#         RandomSwitchingPolicy(exp2_mdp(prm), emp_switch_dist, emp_crit_stop_dist),
#     )
# end

# emp_prms = sobol(N_SOBOL, Box(
#     drift_μ = (-1, 1),
#     noise = (.5, 2.5),
#     drift_σ = (1, 3),
#     threshold = (5, 15),
#     strength_drift_μ = 0,
#     strength_drift_σ = 0.,
#     sample_cost = 0.,
#     switch_cost = 0.,
# ));

# emp_sumstats = compute_sumstats("emp", empirical_policies, emp_prms[JOBS])

# if !CACHE_ONLY
#     emp_prm, emp_ss, tbl, full_loss = minimize_loss(loss, emp_sumstats, emp_prms);
#     show(select(tbl, Not([:strength_drift_μ, :strength_drift_σ]))[1:10, :])
#     let
#         df = simulate_exp2(empirical_policies, emp_prm)
#         trials = make_trials(df); fixations = make_fixations(df)
#         @show loss(exp2_sumstats(trials, fixations))
#         CSV.write("results/exp2/empirical_trials.csv", trials)
#         CSV.write("results/exp2/empirical_fixations.csv", fixations)
#     end
# end
