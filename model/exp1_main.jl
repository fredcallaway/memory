
@everywhere include("common.jl")
@everywhere include("exp1_base.jl")
mkpath("results/exp1")
mkpath("tmp")

N_SOBOL = 50_000
RUN = "jun14"

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

# %% ==================== fitting pipeline ====================

get_simdir(name) = "results/$(RUN)_exp1/simulations/$(name)_trials"

function fit_exp1_model(name, make_policies, box; n_init=N_SOBOL, n_top=cld(n_init, 10), n_sim_top=1_000_000)
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
        sim = simulate_exp1(_policies, prm, n_sim_top)
        sim.rt = sim.rt .+ rand(ndt, nrow(sim))
        CSV.write("$simdir/$i.csv", sim)
    end

    serialize("$fitdir/top", top_tbl)
    top_tbl
end

function simulate_one(name, make_policies, prm)
    sim = simulate_exp1(make_policies, prm)
    ndt = Gamma(prm.α_ndt, prm.θ_ndt)
    loss = likelihood(make_hist(sim), human_hist, Gamma(prm.α_ndt, prm.θ_ndt))
    sim.rt = sim.rt .+ rand(ndt, nrow(sim))
    simdir = get_simdir(name)
    mkpath(simdir)
    CSV.write("$simdir/1.csv", sim)

    tbl = DataFrame([prm])
    tbl.loss = [loss]
    path = "results/$(RUN)_exp1/fits/$name/top"
    mkpath(dirname(path))
    serialize(path, tbl)
end

# %% ==================== optimal ====================

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
opt_prm = NamedTuple(first(eachrow(deserialize("results/$(RUN)_exp1/fits/optimal/top"))))

# %% ==================== empirical ====================

@everywhere function empirical_policies(prm)
    pretest_dist = optimize_stopping_model(skip_rt_hist(human_pretest), prm.α_ndt, prm.θ_ndt)
    crit_dist = optimize_stopping_model(skip_rt_hist(human_trials), prm.α_ndt, prm.θ_ndt)
    (
        RandomStoppingPolicy(pretest_mdp(prm), pretest_dist),
        RandomStoppingPolicy(exp1_mdp(prm), crit_dist)
    )
end

empirical_box = modify(optimal_box; 
    sample_cost = 0,
    αθ_ndt = (100, 1500, :log),
    α_ndt = (1, 100, :log)
)
empirical_tbl = fit_exp1_model("empirical", empirical_policies, empirical_box)

# again with NDT fixed to optimal's value
empirical_fixndt_box = modify(optimal_box; sample_cost=0, opt_prm.α_ndt, opt_prm.θ_ndt)
empirical_fixndt_tbl = fit_exp1_model("empirical_fixndt", empirical_policies, empirical_fixndt_box)

# all parameters fixed
simulate_one("empirical_fixall", empirical_policies, opt_prm)

# %% ==================== flexible stopping ====================

@everywhere begin
    flexible_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), Gamma(prm.α_stop, prm.θ_stop)),
        RandomStoppingPolicy(exp1_mdp(prm), Gamma(prm.α_stop, prm.θ_stop)),
    )
end

flexible_box = modify(optimal_box, 
    sample_cost=0,
    αθ_stop = (1, 20),
    α_stop = (1, 100, :log),
)
fit_exp1_model("flexible", flexible_policies, flexible_box)

flexible_fixndt_box = modify(flexible_box; opt_prm.α_ndt, opt_prm.θ_ndt)
fit_exp1_model("flexible_ndt", flexible_policies, flexible_fixndt_box)

# %% ==================== old empirical ====================

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


# %% ==================== flexible fit effects ====================

@everywhere include("exp1_fit_effects.jl")

full_flex_box = modify(flexible_box,
    αθ_ndt = (100, 1500, :log),
    α_ndt = (1, 100, :log)
)

prms = sample_params(full_flex_box, 100_000)
effects = compute_effects("flexible", flexible_policies, prms)

function score(ef, effect)
    ismissing(ef) && return 0.
    (.1 ≤ ef.accuracy ≤ .9) || return 0.

    ci = getfield(ef, effect)[2]
    fillnan(ci[1])
end

score(effects::Vector, effect) = map(ef->score(ef, effect), effects)

sc, i = findmax(score(effects, :empty_judgement))
sc, i = findmax(score(effects, :empty_pretest))


function top_score(name, make_policies, effects, effect)
    top = partialsortperm(-score(effects, effect), 1:100)
    top_prms = prms[top]
    top_effects = compute_effects(name, make_policies, top_prms; N=1_000_000)
    sc, i = findmax(score(top_effects, effect))
    getfield(top_effects[i], effect), top[i]
end

sc, i = top_score("flexible", flexible_policies, effects, :empty_judgement)
sc, i = top_score("flexible", flexible_policies, effects, :empty_pretest)


