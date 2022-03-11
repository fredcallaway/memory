if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

@everywhere include("common.jl")
@everywhere include("exp2_base.jl")
mkpath("results/exp2")

N_SOBOL = 10_000
RUN = "witherr"
if isempty(ARGS) 
    CACHE_ONLY = false
    JOBS = 1:N_SOBOL
else  # precomputing with batch compute
    CACHE_ONLY = true
    JOBS = let
        chunk_size = 40
        range(chunk_size*parse(Int, ARGS[1]), length=chunk_size)
    end
end

# %% --------
function compute_sumstats(name, make_policies, prms; run=RUN, read_only = false)
    dir = "cache/exp2_$(name)_sumstats_$(run)"
    mkpath(dir)
    map = read_only ? asyncmap : pmap
    @showprogress map(prms) do prm
        cache("$dir/$(stringify(prm))") do
            df = simulate_exp2(make_policies, prm)
            x = exp2_sumstats(make_trials(df), make_fixations(df))
            GC.gc()
            x
        end
    end;
end


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

target = exp2_sumstats(trials, fixations);

cutoff = @chain trials begin
    @rsubset :response_type != "timeout" && :response_type != "intrusion"
    @with quantile(:rt, .95)
    fld(200)
    Int
end

full_timecourse(ss) = @chain ss.unrolled begin
    @orderby :pretest_accuracy_first, :pretest_accuracy_second
    @with combinedims(:timecourse)[1:cutoff, : , :]
end

function loss(ss)
    mae(full_timecourse(target), full_timecourse(ss))
    # ss.accuracy > .8 || return Inf  #
    # (ismissing(ss) || ismissing(ss.unrolled)) && return Inf
    # l = mae(target.unrolled(time= <(cutoff)), ss.unrolled(time= <(cutoff)))
    # ismissing(l) || isnan(l) ? Inf : l
end

# %% ==================== optimal ====================
println("--- optimal ---")

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp2_mdp(prm)),
)

opt_prms = sobol(N_SOBOL, Box(
    drift_μ = (0.5, 1),
    noise = (.5, 2.5),
    drift_σ = (1, 3),
    threshold = (5, 15),
    sample_cost = (0, .01),
    switch_cost = (0, .03),
    strength_drift_μ = 0,
    strength_drift_σ = (0., 1.),
    judgement_noise=1,
))
# opt_sumstats = compute_sumstats("opt", optimal_policies, opt_prms[JOBS], read_only=true);

htoopt_sumstats = compute_sumstats("opt", optimal_policies, opt_prms);

# %% --------
if !CACHE_ONLY
    opt_prm, opt_ss, tbl, full_loss = minimize_loss(loss, opt_sumstats, opt_prms);
    show(select(tbl, Not([:strength_drift_μ, :strength_drift_σ]))[1:10, :])
    let
        df = simulate_exp2(optimal_policies, opt_prm)
        trials = make_trials(df); fixations = make_fixations(df)
        @show loss(exp2_sumstats(trials, fixations))
        CSV.write("results/exp2/optimal_trials.csv", trials)
        CSV.write("results/exp2/optimal_fixations.csv", fixations)
    end
end

# %% ==================== empirical ====================
println("--- empirical ---")

@everywhere begin
    plausible_skips(x) = @rsubset(x, :response_type in ["other", "empty"])
    const emp_pretest_stop_dist = empirical_distribution(plausible_skips(pretest).rt)
    const emp_crit_stop_dist = empirical_distribution(plausible_skips(trials).rt)
    const emp_switch_dist = empirical_distribution(fixations.duration)

    empirical_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
        RandomSwitchingPolicy(exp2_mdp(prm), emp_switch_dist, emp_crit_stop_dist),
    )
end

emp_prms = sobol(N_SOBOL, Box(
    drift_μ = (-1, 1),
    noise = (.5, 2.5),
    drift_σ = (1, 3),
    threshold = (5, 15),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    sample_cost = 0.,
    switch_cost = 0.,
));

emp_sumstats = compute_sumstats("emp", empirical_policies, emp_prms[JOBS])

if !CACHE_ONLY
    emp_prm, emp_ss, tbl, full_loss = minimize_loss(loss, emp_sumstats, emp_prms);
    show(select(tbl, Not([:strength_drift_μ, :strength_drift_σ]))[1:10, :])
    let
        df = simulate_exp2(empirical_policies, emp_prm)
        trials = make_trials(df); fixations = make_fixations(df)
        @show loss(exp2_sumstats(trials, fixations))
        CSV.write("results/exp2/empirical_trials.csv", trials)
        CSV.write("results/exp2/empirical_fixations.csv", fixations)
    end
end
