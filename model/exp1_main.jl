
@everywhere include("common.jl")
@everywhere include("exp1_base.jl")
mkpath("results/exp1")
mkpath("tmp")

N_SOBOL = 50_000
RUN = "may25"

print_header("beginning run $RUN")

if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

# %% ==================== load data ====================

human_pretest = load_data("exp1/pretest")
human_trials = load_data("exp1/trials")
filter!(t-> !ismissing(t.rt), human_pretest)
filter!(t-> !ismissing(t.rt), human_trials)
target = exp1_sumstats(human_trials);

@everywhere human_trials = $human_trials
@everywhere human_pretest = $human_pretest
@everywhere target = $target

# %% ==================== fitting pipeline ====================

get_simdir(name) = "results/$(RUN)_exp1/simulations/$(name)_trials"

function fit_exp1_model(name, make_policies, box; n_init=N_SOBOL, n_top=cld(n_init, 10), n_sim_top=1_000_000)
    print_header(name)
    fitdir = "results/$(RUN)_exp1/fits/$name/"
    mkpath(fitdir)

    prms = sample_params(box, n_init)

    sumstats = compute_sumstats(name, make_policies, prms);
    tbl = compute_loss(sumstats, prms)
    serialize("$fitdir/full", tbl)

    top_prms = map(NamedTuple, eachrow(tbl[1:n_top, :]));
    top_sumstats = compute_sumstats(name, make_policies, top_prms; N=n_sim_top);
    top_tbl = compute_loss(top_sumstats, top_prms)
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

# %% ==================== empirical gamma ====================

@everywhere function empirical_gamma_policies(prm)
    pretest_gamma = optimize_stopping_model(human_pretest, prm.α_ndt, prm.θ_ndt)
    crit_gamma = optimize_stopping_model(human_trials, prm.α_ndt, prm.θ_ndt)
    (
        RandomStoppingPolicy(pretest_mdp(prm), pretest_gamma),
        RandomStoppingPolicy(exp1_mdp(prm), crit_gamma)
    )
end

empirical_gamma_box = modify(optimal_box; 
    sample_cost = 0,
    αθ_ndt = (100, 1500, :log),
    α_ndt = (1, 100, :log)
)

empirical_gamma_tbl = fit_exp1_model("empirical_gamma", empirical_gamma_policies, empirical_gamma_box)

# %% --------
empirical_gamma_box = modify(optimal_box; sample_cost=0, opt_prm.α_ndt, opt_prm.θ_ndt)
empirical_gamma_tbl = fit_exp1_model("empirical_gamma_fixndt", empirical_gamma_policies, empirical_gamma_box)

serialize("results/$(RUN)_exp1/fits/empirical/pretest_gamma", pretest_gamma)

# %% ==================== empirical lesioned ====================

let
    prm = opt_prm
    sim = simulate_exp1(empirical_gamma_policies, prm)
    sim.rt = sim.rt .+ rand(Gamma(prm.α_ndt, prm.θ_ndt), nrow(sim))
    simdir = get_simdir("empirical_lesioned")
    mkpath(simdir)
    CSV.write("$simdir/1.csv", sim)
end


# %% ==================== old empirical ====================

@everywhere begin
    @isdefined(emp_pretest_stop_dist) || const emp_pretest_stop_dist = empirical_distribution(@subset(human_pretest, :response_type .== "empty").rt)
    @isdefined(emp_crit_stop_dist) || const emp_crit_stop_dist = empirical_distribution(@subset(human_trials, :response_type .== "empty").rt)

    old_empirical_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
        RandomStoppingPolicy(exp1_mdp(prm), emp_crit_stop_dist),
    )
end

empirical_box = modify(optimal_box, sample_cost=0)
empirical_tbl = fit_exp1_model("empirical_old", old_empirical_policies, empirical_box)

# # %% ==================== decision bound ====================


# @everywhere bound_policies(prm) = (
#     ConstantBoundPolicy(pretest_mdp(prm), prm.θ),
#     ConstantBoundPolicy(exp1_mdp(prm), prm.θ),
# )

# bound_prms = sobol(N_SOBOL, Box(
#     drift_μ = (-0.5, 0.5),
#     noise = (0.1, 2.),
#     drift_σ = (0.5, 2),
#     threshold = (2, 20),
#     sample_cost = 0.,
#     θ = (1, 15),
#     τ = (.001, 1, :log),
#     strength_noise=(0,1),
#     judgement_noise=1,
# ))

# bound_sumstats = fit_exp1_model("bound", bound_policies, bound_prms);

# println("Done!")
