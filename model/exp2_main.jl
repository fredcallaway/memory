if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

RUN = "may25"
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
end

human_trials_witherr = @chain human_trials_witherr begin
    @rsubset :n_pres > 0 && !ismissing(:rt)
end

@everywhere human_trials = $human_trials
@everywhere human_trials_witherr = $human_trials_witherr
@everywhere human_pretest = $human_pretest
@everywhere human_fixations = $human_fixations

const ss_human = exp2_sumstats(human_trials, human_fixations);

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

    @showprogress "simulate" pmap(enumerate(prms)) do (i, prm)
        pre_pol, crit_pol = make_policies(prm)

        if ndt == :optimize
            sim = simulate_exp2(pre_pol, crit_pol; prm.within_σ, prm.between_σ)
            res = optimize_duration_noise(sim, human_fixations)
            α_ndt, θ_ndt = res.minimizer
            serialize(fitpath, (;prm..., α_ndt, θ_ndt))
            ndt = Gamma(α_ndt, θ_ndt)
        end

        sim = simulate_exp2(pre_pol, crit_pol; prm.within_σ, prm.between_σ)
        add_duration_noise!(sim, ndt)

        trials = make_trials(sim); fixations = make_fixations(sim)
        CSV.write("$trialdir/$i.csv", trials)
        CSV.write("$fixdir/$i.csv", fixations)
        ss = exp2_sumstats(trials, fixations)
        ss
        # (;ss..., res)
    end
end

# %% ==================== optimal ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp2_mdp(prm)),
)

optimal_results = write_sims("optimal", optimal_policies)
serialize("tmp/$(RUN)_exp2_optimal_results", optimal_results)

# %% ==================== empirical gamma ====================

function optimize_switching_model(α_ndt, θ_ndt)
    switch_dist = @chain human_fixations begin
        DataFrames.rename(:duration => :rt)
        @rtransform :response_type = "empty"  # a hack to prevent the filtering in optimize_stopping_model
        @rsubset :presentation ≠ :n_pres
        optimize_stopping_model(α_ndt, θ_ndt)
    end
end

function optimize_stopping_model_exp2(α_ndt, θ_ndt)
    # we have to split by number of fixations becausee the
    # total NDT is the sum of NDT on each fixation

    dt = MS_PER_SAMPLE; maxt = MAX_TIME

    human = @chain human_trials_witherr begin
        @rsubset :response_type == "empty" && 1 ≤ :n_pres ≤ 5
        @rtransform :rt = quantize(:total_first + :total_second, dt)
        wrap_counts(rt = dt:dt:maxt, n_pres=1:5)
        normalize!
    end
    wts = ssum(human, :rt)

    ndt_only = copy(human)
    for n_pres in 1:5
        ndt_dist = Gamma(α_ndt, n_pres * θ_ndt)
        ndt_only[:, n_pres] .= wts[n_pres] .* diff([0; cdf(ndt_dist, ndt_only.rt)])
    end

    α, θ = optimize_ndt(ndt_only, human).minimizer
    # α, θ = optimize_ndt(ndt_only, human).minimizer
    Gamma(α, θ / MS_PER_SAMPLE)  # convert to units of samples
end

(;α_ndt, θ_ndt) = deserialize("results/$(RUN)_exp2/fits/optimal")
@isdefined(pretest_stopping) || const pretest_stopping = deserialize("results/$(RUN)_exp1/fits/empirical/pretest_gamma")
@isdefined(crit_switching) || const crit_switching = optimize_switching_model(α_ndt, θ_ndt)
@isdefined(crit_stopping) || const crit_stopping = optimize_stopping_model_exp2(α_ndt, θ_ndt)

@everywhere begin
    @isdefined(pretest_stopping) || const pretest_stopping = $pretest_stopping
    @isdefined(crit_switching) || const crit_switching = $crit_switching
    @isdefined(crit_stopping) || const crit_stopping = $crit_stopping

    empirical_gamma_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), pretest_stopping),
        RandomSwitchingPolicy(exp2_mdp(prm), crit_switching, crit_stopping),
    )
end

# need to use the optimal NDT parameters ✔ -> need to check results
write_sims("empirical_gamma", empirical_gamma_policies; ndt=Gamma(α_ndt, θ_ndt))

# %% ==================== empirical ====================

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
empirical_results = write_sims("empirical", empirical_policies)


