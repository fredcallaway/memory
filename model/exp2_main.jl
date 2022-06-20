if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

RUN = "jun14"
@everywhere include("common.jl")
@everywhere include("exp2_base.jl")
mkpath("results/$(RUN)_exp2")

# %% ==================== load data ====================

human_pretest = CSV.read("../data/processed/exp2/pretest.csv", DataFrame, missingstring="NA")
human_trials = CSV.read("../data/processed/exp2/trials.csv", DataFrame, missingstring="NA")
human_trials_witherr = CSV.read("../data/processed/exp2/trials_witherr.csv", DataFrame, missingstring="NA")
human_fixations = CSV.read("../data/processed/exp2/fixations.csv", DataFrame, missingstring="NA")

human_pretest = @rsubset human_pretest :practice == false :block == 3
human_trials = @chain human_trials begin
    @rsubset :n_pres > 0
    @rsubset :response_type != "intrusion"
    @rtransform :choose_first = :response_type == "correct" ? :choose_first : missing
    @rtransform :rt = :total_first + :total_second
    select(Not(:raw_rt))
end

human_trials_witherr = @chain human_trials_witherr begin
    @rsubset :n_pres > 0 && !ismissing(:rt)
    @rtransform :choose_first = :response_type == "correct" ? :choose_first : missing
    @rtransform :rt = :total_first + :total_second
end

@everywhere human_trials = $human_trials
@everywhere human_trials_witherr = $human_trials_witherr
@everywhere human_pretest = $human_pretest
@everywhere human_fixations = $human_fixations

function write_sims(name, make_policies; ndt=:optimize, n_top=1)
    top_table = deserialize("results/$(RUN)_exp1/fits/$name/top")
    exp1_top = eachrow(top_table)[1:n_top]
    prms = map(exp1_top) do prm
        (;prm..., switch_cost=prm.sample_cost)
    end

    trialdir = "results/$(RUN)_exp2/simulations/$(name)_trials"
    fixdir = "results/$(RUN)_exp2/simulations/$(name)_fixations"
    fitpath = "results/$(RUN)_exp2/fits/$name"
    mkpath(trialdir)
    mkpath(fixdir)
    mkpath(dirname(fitpath))

    map(enumerate(prms)) do (i, prm)
        pre_pol, crit_pol = make_policies(prm)

        if ndt == :optimize
            println("optimizing ndt")
            sim = simulate_exp2(pre_pol, crit_pol; prm.within_σ, prm.between_σ)
            res = optimize_duration_noise(sim, human_fixations)
            α_ndt, θ_ndt = res.minimizer
            ndt = Gamma(α_ndt, θ_ndt)
        end
        serialize(fitpath, (;prm..., α_ndt=ndt.α, θ_ndt=ndt.θ))

        sim = simulate_exp2(pre_pol, crit_pol; prm.within_σ, prm.between_σ)
        add_duration_noise!(sim, ndt)

        trials = make_trials(sim); fixations = make_fixations(sim)
        CSV.write("$trialdir/$i.csv", trials)
        CSV.write("$fixdir/$i.csv", fixations)

    end
end

# %% ==================== optimal ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp2_mdp(prm)),
)

optimal_results = write_sims("optimal", optimal_policies)
serialize("tmp/$(RUN)_exp2_optimal_results", optimal_results)

# %% ==================== empirical ====================

function optimize_switching_model(α_ndt, θ_ndt)
    optimize_stopping_model(duration_hist(human_fixations), prm.α_ndt, prm.θ_ndt)
end

function optimize_stopping_model_exp2(α_ndt, θ_ndt)
    # we have to split by number of fixations because the
    # total NDT is the sum of NDT on each fixation
    dt = MS_PER_SAMPLE; maxt = MAX_TIME

    human = @chain human_trials_witherr begin
        @rsubset :response_type == "empty" && 1 ≤ :n_pres ≤ 5
        @rtransform :rt = quantize(:rt, dt)
        wrap_counts(rt = dt:dt:maxt, n_pres=1:5)
        normalize!
    end
    wts = ssum(human, :rt)

    ndt_only = copy(human)
    for n_pres in 1:5
        ndt_dist = Gamma(α_ndt, n_pres * θ_ndt)
        ndt_only[:, n_pres] .= wts[n_pres] .* diff([0; cdf(ndt_dist, ndt_only.rt)])
    end

    α, θ = optimize_ndt(ssum(ndt_only, :n_pres), ssum(human, :n_pres)).minimizer
    # α, θ = optimize_ndt(ndt_only, human).minimizer
    Gamma(α, θ / MS_PER_SAMPLE)  # convert to units of samples
end

pretest_stopping = let
    exp1_prm = first(eachrow(deserialize("results/$(RUN)_exp1/fits/empirical_fixndt/top")))
    optimize_stopping_model(skip_rt_hist(human_pretest), exp1_prm.α_ndt, exp1_prm.θ_ndt)
end

(;α_ndt, θ_ndt) = deserialize("results/$(RUN)_exp2/fits/optimal")
crit_switching = optimize_stopping_model(duration_hist(human_fixations), α_ndt, θ_ndt)
crit_stopping = optimize_stopping_model_exp2(α_ndt, θ_ndt)


@everywhere begin
    pretest_stopping = $pretest_stopping
    crit_switching = $crit_switching
    crit_stopping = $crit_stopping

    empirical_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), pretest_stopping),
        RandomSwitchingPolicy(exp2_mdp(prm), crit_switching, crit_stopping),
    )
end

write_sims("empirical_fixndt", empirical_policies; ndt=Gamma(α_ndt, θ_ndt))
write_sims("empirical_fixall", empirical_policies; ndt=Gamma(α_ndt, θ_ndt))
write_sims("flexible_ndt", empirical_policies; ndt=Gamma(α_ndt, θ_ndt))
write_sims("flexible", empirical_policies; ndt=Gamma(α_ndt, θ_ndt))

# %% ==================== empirical (old) ====================

@everywhere begin
    plausible_skips(x) = @rsubset(x, :response_type in ["other", "empty"])
    const emp_pretest_stop_dist = empirical_distribution(plausible_skips(human_pretest).rt)
    const emp_crit_stop_dist = empirical_distribution(skipmissing(plausible_skips(human_trials_witherr).rt))
    const emp_switch_dist = empirical_distribution(human_fixations.duration)

    empirical_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
        RandomSwitchingPolicy(exp2_mdp(prm), emp_switch_dist, emp_crit_stop_dist),
    )
end
empirical_results = write_sims("empirical_old", empirical_policies)


# %% ==================== flexible ====================

@everywhere flexible_policies(prm) = (
    RandomStoppingPolicy(pretest_mdp(prm), Gamma(prm.α_stop, prm.θ_stop)),
    RandomSwitchingPolicy(exp2_mdp(prm), Gamma(prm.α_switch, prm.θ_switch), Gamma(prm.α_stop, prm.θ_stop)),
)
@everywhere include("exp2_fitting.jl")
flex_box = Box(
    drift_μ = (-0.5, 0.5),
    between_σ = (0, 1),
    noise = (0, 1),
    threshold = 1,
    sample_cost = 0,
    switch_cost = 0,
    within_σ=0,
    judgement_noise=0.1,
    αθ_stop = (1, 20),
    α_stop = (1, 100, :log),
    αθ_switch = (1, 20),
    α_switch = (1, 100, :log),
    αθ_ndt = (100, 1500, :log),
    α_ndt = (1, 100, :log)
)

prms = sample_params(flex_box, 10_000)
sumstats = compute_sumstats("flexible", flexible_policies, prms; read_only=true)

# %% --------
fillnan(x, repl=0.) = isnan(x) ? repl : x
ok_sumstats = filter(sumstats) do ss
    ss.accuracy > 0.5 &&
    sum((!isnan).(ss.nonfixated)) == 3 &&
    sum((!isnan).(ss.fixated)) == 3 &&
    sum((!isnan).(ss.prop_first)) == 5 &&
    true
end

fixated_effect(ss) = ss.fixated[2] - ss.fixated[1]
nonfixated_effect(ss) = ss.nonfixated[1] - ss.nonfixated[3]
final_effect(ss) = ss.final[2] - ss.final[1]
prop_first_effect(ss) = ss.prop_first[5] - ss.prop_first[1]


ss_human = exp2_sumstats(human_trials, human_fixations)



fixated_effect(ss_human)
ss_human.nonfixated
prop_first_effect(ss_human)

argmax(ok_sumstats) do ss
    fixated_effect(ss)

ss.nonfixated
ss.fixated


ss.nonfixated
ss.fixated

ss.final


# %% --------

function fit_exp2_model(name, make_policies, box; n_init=N_SOBOL, n_top=cld(n_init, 10), n_sim_top=1_000_000)
    print_header(name)
    fitdir = "results/$(RUN)_exp1/fits/$name/"
    mkpath(fitdir)

    prms = sample_params(box, n_init)
    hists = compute_histograms(name, make_policies, prms);
    tbl = compute_loss(hists, prms)
    serialize("$fitdir/full", tbl)

    top_prms = map(NamedTuple, eachrow(tbl[1:n_top, :]));
    top_hists = compute_histograms(name, make_policies, top_prms; N=n_sim_top);
    top_tbl = compute_loss(top_hists, top_prms)
    top_tbl.judgement_noise = 0.5 .* top_tbl.drift_σ
    display(top_tbl[1:13, :])

    simdir = get_simdir(name)
    mkpath(simdir)
    @showprogress "simulating" pmap(enumerate(eachrow(top_tbl)[1:5])) do (i, row)
        ndt = Gamma(row.α_ndt, row.θ_ndt)
        prm = NamedTuple(row)
        sim = simulate_exp1(make_policies, prm, n_sim_top)
        sim.rt = sim.rt .+ rand(ndt, nrow(sim))
        CSV.write("$simdir/$i.csv", sim)
    end

    serialize("$fitdir/top", top_tbl)
    top_tbl
end






