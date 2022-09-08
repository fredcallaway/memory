if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

@everywhere begin
    EXP1_RUN = "sep7_exp1"
    RUN = "sep7_exp2"
    include("common.jl")
    include("exp2_base.jl")
    N_SOBOL = 50_000
end

mkpath("results/$(RUN)")

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

@everywhere human_hist = make_hist(human_fixations)

@everywhere function write_sims(name, make_policies; ndt=:optimize, n_top=1)
    top_table = deserialize("results/$(EXP1_RUN)/fits/$name/top")
    exp1_top = eachrow(top_table)[1:n_top]
    prms = map(exp1_top) do prm
        (;prm..., switch_cost=prm.sample_cost)
    end

    trialdir = "results/$(RUN)/simulations/fixed_$(name)_trials"
    fixdir = "results/$(RUN)/simulations/fixed_$(name)_fixations"
    fitpath = "results/$(RUN)/fits/fixed_$name"
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
            @show (;α_ndt, θ_ndt)
            ndt_dist = Gamma(α_ndt, θ_ndt)
        else
            ndt_dist = ndt  # note: untested (possibe variable scope issue)
        end
        serialize(fitpath, (;prm..., α_ndt=ndt_dist.α, θ_ndt=ndt_dist.θ))

        sim = simulate_exp2(pre_pol, crit_pol; prm.within_σ, prm.between_σ)
        add_duration_noise!(sim, ndt_dist)

        trials = make_trials(sim); fixations = make_fixations(sim)
        CSV.write("$trialdir/$i.csv", trials)
        CSV.write("$fixdir/$i.csv", fixations)
    end
    println("Wrote $trialdir")
end

function fit_model(name, make_policies, box; n_init=N_SOBOL, n_top=cld(n_init, 10), n_sim_top=1_000_000)
    print_header(name)
    fitdir = "results/$(RUN)/fits/$name/"
    mkpath(fitdir)

    prms = sample_params(box, n_init)
    hists = compute_histograms(name, make_policies, prms);
    name == "optimal" && GC.gc()  # minimimize memory usage
    tbl = compute_loss(hists, prms)

    serialize("$fitdir/full", tbl)

    top_prms = map(NamedTuple, eachrow(tbl[1:n_top, :]));
    top_hists = compute_histograms(name, make_policies, top_prms; N=n_sim_top);
    name == "optimal" && GC.gc()  # minimimize memory usage
    top_tbl = compute_loss(top_hists, top_prms)
    display(top_tbl[1:10, :])
    serialize("$fitdir/top", top_tbl)
    
    trialdir = "results/$(RUN)/simulations/$(name)_trials"
    fixdir = "results/$(RUN)/simulations/$(name)_fixations"
    mkpath(trialdir)
    mkpath(fixdir)

    @showprogress "simulating" pmap(enumerate(eachrow(top_tbl)[1:1])) do (i, row)
        prm = NamedTuple(row)
        sim = simulate_exp2(make_policies, prm)
        ndt = Gamma(prm.α_ndt, prm.θ_ndt)
        add_duration_noise!(sim, ndt)

        trials = make_trials(sim); fixations = make_fixations(sim)
        CSV.write("$trialdir/$i.csv", trials)
        CSV.write("$fixdir/$i.csv", fixations)
    end
    top_tbl
end


# %% ==================== optimal ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp2_mdp(prm)),
)

@spawn write_sims("optimal", optimal_policies)

# optimal_box = Box(
#     drift_μ = (-0.1, 0.1),
#     noise = (0, 0.3),
#     threshold = 1,
#     sample_cost = (0, .03),
#     switch_cost = (0, .1),
#     between_σ = (0, .4),
#     within_σ=0,
#     judgement_noise=0,
# )

# fit_model("optimal", optimal_policies, optimal_box; n_init=5_000)

# exp1_fit = load_fit("optimal", EXP1_RUN)

# constrained_optimal_box = Box(
#     drift_μ = (exp1_fit.drift_µ, 0.2),
#     noise = exp1_fit.noise,
#     threshold = 1,
#     sample_cost = exp1_fit.sample_cost,
#     switch_cost = (0, .05),
#     between_σ = (0, 2 * exp1_fit.between_σ),
#     within_σ=0,
#     judgement_noise=0,
# )

# fit_model("constrained_optimal", optimal_policies, constrained_optimal_box; n_init=5_000)

# if I want to take out the presentation dimension
# hists = compute_histograms("optimal", optimal_policies, sample_params(optimal_box, 5000); read_only=true)

# %% ==================== flexible ====================

flexible_box = Box(
    drift_μ = (-0.5, 0.5),
    between_σ = (0, 1),
    noise = (0, 1),
    threshold = 1,
    sample_cost = 0,
    switch_cost = 0,
    within_σ = 0,
    αθ_stop = (1, 50),
    α_stop = (1, 100, :log),
    αθ_switch = (1, 50),
    α_switch = (.01, 1, :log),
)

@everywhere flexible_policies(prm) = (
    RandomStoppingPolicy(pretest_mdp(prm), Gamma(prm.α_stop, prm.θ_stop)),
    RandomSwitchingPolicy(exp2_mdp(prm), Gamma(prm.α_switch, prm.θ_switch), Gamma(prm.α_stop, prm.θ_stop)),
)

fit_model("flexible", flexible_policies, flexible_box)

# exp1_fit = @chain deserialize("results/$(EXP1_RUN)/fits/flexible/top") begin
#     select([:drift_μ, :between_σ, :noise])
#     eachrow
#     first
# end

# fit_model("fixed_flexible", flexible_policies, modify(flexible_box; exp1_fit...))
# %% --------
# @chain deserialize("results/$(RUN)/fits/fixed_flexible/top") select(Not([:threshold, :sample_cost, :switch_cost, :within_σ]))
# @chain deserialize("results/$(RUN)/fits/flexible/top") select(Not([:threshold, :sample_cost, :switch_cost, :within_σ]))


# %% ==================== empirical (old) ====================

@everywhere begin
    # plausible_skips(x) = @rsubset(x, :response_type in ["other", "empty"])
    plausible_skips(x) = @rsubset(x, :response_type == "empty")
    const emp_pretest_stop_dist = empirical_distribution(plausible_skips(human_pretest).rt)
    const emp_crit_stop_dist = empirical_distribution(skipmissing(plausible_skips(human_trials_witherr).rt))
    const emp_switch_dist = empirical_distribution(human_fixations.duration)

    old_empirical_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
        RandomSwitchingPolicy(exp2_mdp(prm), emp_switch_dist, emp_crit_stop_dist),
    )
end

@spawn write_sims("empirical_old", old_empirical_policies)


# %% ==================== report parameters ====================

x = deserialize("results/$RUN/fits/fixed_optimal")
write_tex("mle_fixed_optimal", "\\(
    \\mu_\\text{NDT} = $(fmt(0, x.α_ndt * x.θ_ndt)),\\ 
    \\alpha_\\text{NDT} = $(fmt(2, x.α_ndt))
\\)")

# x = load_fit("optimal")
# write_tex("optimal", "\\(
#     \\mu_0 = $(fmt(3, x.drift_µ)),\\ 
#     \\sigma_0 = $(fmt(3, x.drift_σ)),\\ 
#     \\sigma_x = $(fmt(3, x.noise)),\\ 
#     \\samplecost = $(fmt(3, x.sample_cost)),\\ 
#     \\mu_\\text{NDT} = $(fmt(0, x.α_ndt * x.θ_ndt)),\\ 
#     \\alpha_\\text{NDT} = $(fmt(2, x.α_ndt))
# \\)")


x = load_fit("flexible")
write_tex("mle_flexible", "\\(
    \\mu_0 = $(fmt(3, x.drift_µ)),\\ 
    \\sigma_0 = $(fmt(3, x.drift_σ)),\\ 
    \\sigma_x = $(fmt(3, x.noise)),\\ 
    \\mu_\\text{stop} = $(fmt(0, MS_PER_SAMPLE * x.αθ_stop)),\\ 
    \\alpha_\\text{stop} = $(fmt(2, x.α_stop)),\\ 
    \\mu_\\text{switch} = $(fmt(0, MS_PER_SAMPLE * x.αθ_switch)),\\ 
    \\alpha_\\text{switch} = $(fmt(2, x.α_switch)),\\ 
    \\mu_\\text{NDT} = $(fmt(0, x.α_ndt * x.θ_ndt)),\\ 
    \\alpha_\\text{NDT} = $(fmt(2, x.α_ndt)),\\ 
\\)")

# for name in ["optimal", "flexible"]
#     x = load_xs(name)
#     open("results/xs/exp2/nll_$name", "w") do f
#         writev(f, round(Int, x.loss * nrow(human_trials)))
#     end
# end

# %% ==================== can the null model get the effects? ====================

@everywhere include("exp2_fit_effects.jl")

prop_nfix(ef, num) = sum(ef.nfix[num:end]) / sum(ef.nfix)
rt_µ, rt_σ = juxt(mean, std)(human_trials.rt)
acc = mean(human_trials_witherr.response_type .== "correct")

function reasonable_wrapper(f)
    function wrapped(ef)
        ismissing(ef) && return -Inf
        # abs(ef.accuracy - acc) ≤ 0.5 || return -Inf
        0.01 < ef.accuracy < 0.99 || return -Inf
        # .05 ≤ ef.accuracy ≤ 0.95  || return -Inf
        # 0.1 ≤ ef.accuracy || return -Inf
        prop_nfix(ef, 2) > .01 || return -Inf
        f(ef)
    end
end

function top_score(score_fn, name, make_policies, prms, effects; double_check=true)
    score = reasonable_wrapper(score_fn)
    scores = map(score, effects)
    if double_check
        top = partialsortperm(-scores, 1:100)
        top_prms = prms[top]
        top_effects = compute_effects(name, make_policies, top_prms; N=1_000_000)
        sc, i = findmax(map(score, top_effects))
        sc, top_effects[i], top_prms[i]
    else
        sc, i = findmax(scores)
        sc, effects[i], prms[i]
    end
end

max_ndt = @chain human_fixations begin
    @rsubset :presentation != :n_pres
    @with mean(:duration) - MS_PER_SAMPLE
end

flexible_ndt_box = modify(flexible_box,
    αθ_ndt = (100, max_ndt),
    α_ndt = (1, 100, :log)
)

prms = sample_params(flexible_ndt_box, 100_000);
effects = compute_effects("flexible", flexible_policies, prms);

# %% --------

MIN_EFFECT = 5

sc, ef, prm = top_score("flexible", flexible_policies, prms, effects) do ef
    lower_ci(ef, :prop_first)
end
@info "prop_first" ef.prop_first ef.accuracy ef.rt
write_tex("lesion_search/prop_first", fmt_ci(ef.prop_first))
@assert sc > .01

# %% --------

sc, ef, prm = top_score("flexible", flexible_policies, prms, effects) do ef
    lower_ci(ef, :final)
end
@info "final" ef.final ef.accuracy ef.rt
write_tex("lesion_search/final", fmt_ci(ef.final))
@assert sc > MIN_EFFECT
# sim = simulate_exp2(flexible_policies, prm);

# %% --------

sc, ef, prm_fixated = top_score("flexible", flexible_policies, prms, effects) do ef
    lower_ci(ef, :fixated)
end
@info "fixated" ef.fixated ef.nonfixated ef.accuracy ef.rt
write_tex("lesion_search/fixated", fmt_ci(ef.fixated))
write_tex("lesion_search/fixated_accuracy", string(round1(100*ef.accuracy), "\\%"))

sc, ef, prm_nonfixated = top_score("flexible", flexible_policies, prms, effects) do ef
    lower_ci(ef, :nonfixated)
end
@info "nonfixated" ef.fixated ef.nonfixated ef.accuracy ef.rt
write_tex("lesion_search/nonfixated", fmt_ci(ef.nonfixated))
write_tex("lesion_search/nonfixated_accuracy", string(round1(100*ef.accuracy), "\\%"))

@assert prm_fixated == prm_nonfixated

# %% --------

sc, ef, prm = top_score("flexible", flexible_policies, prms, effects) do ef
    ef.accuracy > 0.65 || return -Inf
    lower_ci(ef, :fixated)
end
ef.pretest_accuracy
@info "accurate_fixated" ef.fixated ef.nonfixated ef.accuracy ef.rt
write_tex("lesion_search/accurate_fixated", fmt_ci(ef.fixated, 4))


sc, ef, prm = top_score("flexible", flexible_policies, prms, effects) do ef
    ef.accuracy > 0.65 || return -Inf
    lower_ci(ef, :nonfixated)
end
@info "accurate_nonfixated" ef.fixated ef.nonfixated ef.accuracy ef.rt
write_tex("lesion_search/accurate_nonfixated", fmt_ci(ef.nonfixated, 4))
