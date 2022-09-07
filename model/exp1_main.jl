
@everywhere include("common.jl")
@everywhere include("exp1_base.jl")
mkpath("results/exp1")
mkpath("tmp")

N_SOBOL = 50_000
RUN = "aug16_exp1"

print_header("beginning run $RUN")

if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

# %% ==================== load data ====================

human_pretest = load_data("exp1/pretest")
human_trials = load_data("exp1/trials")
filter!(t-> !ismissing(t.rt), human_pretest)
filter!(t-> !ismissing(t.rt), human_trials)
human_hist = make_hist(human_trials);

@everywhere human_trials = $human_trials
@everywhere human_pretest = $human_pretest
# @everywhere human_hist = $human_hist

NO_RUN = true


# %% ==================== fitting pipeline ====================

get_simdir(name) = "results/$(RUN)/simulations/$(name)_trials"

function fit_exp1_model(name, make_policies, box; n_init=N_SOBOL, n_top=cld(n_init, 10), n_sim_top=1_000_000)
    NO_RUN && return
    print_header(name)
    fitdir = "results/$(RUN)/fits/$name/"
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
    @showprogress "simulating" pmap(enumerate(eachrow(top_tbl)[1:3])) do (i, row)
        ndt = Gamma(row.α_ndt, row.θ_ndt)
        prm = NamedTuple(row)
        sim = simulate_exp1(make_policies, prm, n_sim_top)
        sim.rt = sim.rt .+ rand(ndt, nrow(sim))
        CSV.write("$simdir/$i.csv", sim)
    end

    serialize("$fitdir/top", top_tbl)
    top_tbl
end


# %% ==================== optimal model ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp1_mdp(prm)),
)

optimal_box = Box(
    drift_μ = (-0.5, 0.5),
    noise = (0, 1),
    threshold = 1,
    sample_cost = (0, .05),
    between_σ = (0, 1),
    within_σ=0,
    judgement_noise=0.1,
)

optimal_tbl = fit_exp1_model("optimal", optimal_policies, optimal_box)
opt_prm = NamedTuple(first(eachrow(deserialize("results/$(RUN)/fits/optimal/top"))))


# %% ==================== new flexible null model ====================

@everywhere begin
    flexible_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), Gamma(prm.α_stop, prm.θ_stop)),
        RandomStoppingPolicy(exp1_mdp(prm), Gamma(prm.α_stop, prm.θ_stop)),
    )
end

flexible_box = modify(optimal_box, 
    sample_cost=0,
    αθ_stop = (1, 50, :log),
    α_stop = (1, 100, :log),
)
fit_exp1_model("flexible", flexible_policies, flexible_box)


# %% ==================== old empirical null model ====================

@everywhere begin
    @isdefined(emp_pretest_stop_dist) || const emp_pretest_stop_dist = empirical_distribution(@subset(human_pretest, :response_type .== "empty").rt)
    @isdefined(emp_crit_stop_dist) || const emp_crit_stop_dist = empirical_distribution(@subset(human_trials, :response_type .== "empty").rt)

    old_empirical_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
        RandomStoppingPolicy(exp1_mdp(prm), emp_crit_stop_dist),
    )
end

empirical_old_box = modify(optimal_box, sample_cost=0)
empirical_old_tbl = fit_exp1_model("empirical_old", old_empirical_policies, empirical_old_box)

# %% ==================== report parameters ====================
mkpath("results/tex/exp1/")


fit = load_fit("optimal")
write_tex("mle_optimal", "\\(
    \\mu_0 = $(round3(fit.drift_µ)),\\ 
    \\sigma_0 = $(round3(fit.drift_σ)),\\ 
    \\sigma = $(round3(fit.noise)),\\ 
    \\samplecost = $(round3(fit.sample_cost)),\\ 
    \\mu_\\text{NDT} = $(round2(fit.α_ndt * fit.θ_ndt)),\\ 
    \\alpha_\\text{NDT} = $(round2(fit.α_ndt))
\\)")

fit = load_fit("flexible")
write_tex("mle_flexible", "\\(
    \\mu_0 = $(round3(fit.drift_µ)),\\ 
    \\sigma_0 = $(round3(fit.drift_σ)),\\ 
    \\sigma = $(round3(fit.noise)),\\ 
    \\mu_\\text{stop} = $(MS_PER_SAMPLE * round2(fit.αθ_stop)),\\ 
    \\alpha_\\text{stop} = $(round2(fit.α_stop)),\\ 
    \\mu_\\text{NDT} = $(round2(fit.α_ndt * fit.θ_ndt)),\\ 
    \\alpha_\\text{NDT} = $(round2(fit.α_ndt))
\\)")

for name in ["optimal", "flexible", "empirical_old"]
    fit = load_fit(name)
    write_tex("nll_$name", string(round(Int, fit.loss * nrow(human_trials))))
end

# %% ==================== can the null model get the effects? ====================

function reasonable_wrapper(f)
    function wrapped(ef)
        ismissing(ef) && return -Inf
        (.01 ≤ ef.accuracy ≤ 0.99) || return -Inf
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
        @everywhere GC.gc()
        sc, i = findmax(map(score, top_effects))
        sc, top_effects[i], top_prms[i]
    else
        sc, i = findmax(scores)
        sc, effects[i], prms[i]
    end
end

flexible_ndt_box = modify(flexible_box,
    αθ_ndt = (100, 1500, :log),
    α_ndt = (1, 100, :log)
)

prms = sample_params(flexible_ndt_box, 100_000);
effects = compute_effects("flexible", flexible_policies, prms; read_only=true);

# ---------- correct ---------- #
MIN_EFFECT = 5  # milis

sc, ef, prm = top_score("flexible", flexible_policies, prms, effects) do ef
    lower_ci(ef, :correct_judgement)
end
@info "correct judgement" ef.correct_judgement ef.accuracy
write_tex("lesion_search/correct_judgement", fmt_ci(ef.correct_judgement))
@assert 5sc > MIN_EFFECT  # multiply by 5 to capture full range

sc, ef, prm = top_score("flexible", flexible_policies, prms, effects) do ef
    lower_ci(ef, :correct_pretest)
end
@info "correct pretest" ef.correct_pretest ef.accuracy
write_tex("lesion_search/correct_pretest", fmt_ci(ef.correct_pretest))
@assert sc > MIN_EFFECT

# ---------- empty ---------- #

sc, ef, prm = top_score("flexible", flexible_policies, prms, effects) do ef
    lower_ci(ef, :empty_judgement)
end
@info "empty judgement" ef.empty_judgement ef.correct_judgement ef.accuracy
@assert 5sc > MIN_EFFECT
@assert lower_ci(ef, :correct_judgement) < MIN_EFFECT

write_tex("lesion_search/empty_judgement", fmt_ci(ef.empty_judgement))
write_tex("lesion_search/empty_judgement_correct", fmt_ci(ef.correct_judgement; negate=true))

sc, ef, prm = top_score("flexible", flexible_policies, prms, effects) do ef
    x = lower_ci(ef, :empty_pretest)
    isfinite(x) ? x : -Inf
end
@info "empty pretest" ef.empty_pretest ef.correct_pretest ef.accuracy
@assert sc < MIN_EFFECT
write_tex("lesion_search/empty_pretest", fmt_ci(ef.empty_pretest))

# ---------- crossover ---------- #

sc, ef, prm = top_score("flexible", flexible_policies, prms, effects) do ef
    min(lower_ci(ef, :empty_judgement), lower_ci(ef, :correct_judgement))
end
@info "judgements" ef.empty_judgement ef.correct_judgement ef.accuracy
@assert 5ef.empty_pretest[1] < MIN_EFFECT

# sc ≤ 10 # effectively zero

sc, ef, prm = top_score("flexible", flexible_policies, prms, effects) do ef
    min(lower_ci(ef, :empty_pretest), lower_ci(ef, :correct_pretest))
end
@info "pretest" ef.empty_pretest ef.correct_pretest ef.accuracy
@assert ef.empty_pretest[1] < MIN_EFFECT
# @assert sc ≤ 10 # effectively zero
