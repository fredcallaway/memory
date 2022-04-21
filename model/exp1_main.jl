
@everywhere include("common.jl")
@everywhere include("exp1_base.jl")
mkpath("results/exp1")
mkpath("tmp")

N_SOBOL = 50_000
RUN = "apr18"

if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

# %% ==================== load data ====================

pretest = load_data("exp1/pretest")
trials = load_data("exp1/trials")
target = exp1_sumstats(trials);

@everywhere trials = $trials
@everywhere pretest = $pretest
@everywhere target = $target

# %% ==================== fitting pipeline ====================

function fit_exp1_model(name, make_policies, prms; n_top=5000, n_sim_top=1_000_000)
    sumstats = compute_sumstats(name, make_policies, prms);
    tbl = compute_loss(loss, sumstats, prms);
    serialize("tmp/$(RUN)_exp1_tbl_$name", tbl)
    display(select(tbl, Not([:judgement_noise]))[1:13, :])

    top_prms = map(NamedTuple, eachrow(tbl[1:n_top, :]));
    top_sumstats = compute_sumstats(name, make_policies, top_prms; N=n_sim_top);
    top_tbl = compute_loss(loss, top_sumstats, top_prms)
    display(select(top_tbl, Not([:judgement_noise]))[1:13, :])

    # rt_noise = @showprogress "optimize rt noise" pmap(eachrow(top_tbl)) do row
    #     ismissing(row.ss) && return (α=NaN, θ=NaN)
    #     Gamma(optimize_rt_noise(row.ss).minimizer...)
    # end
    # top_tbl.rt_α = getfield.(rt_noise, :α)
    # top_tbl.rt_θ = getfield.(rt_noise, :θ)

    mkpath("results/$(RUN)_exp1/$(name)_trials/")
    @showprogress "simulating" pmap(enumerate(eachrow(top_tbl)[1:5])) do (i, row)
        # rt_noise = Gamma(row.rt_α, row.rt_θ)
        rt_noise = Gamma(optimize_rt_noise(row.ss).minimizer...)
        prm = NamedTuple(row)
        sim = simulate_exp1(make_policies, prm, n_sim_top)
        sim.rt = sim.rt .+ rand(rt_noise, nrow(sim))
        CSV.write("results/$(RUN)_exp1/$(name)_trials/$i.csv", sim)
    end

    serialize("tmp/$(RUN)_exp1_fits_$name", top_tbl)
    top_tbl
end

# %% ==================== optimal ====================

print_header("optimal")

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp1_mdp(prm)),
)

optimal_prms = sample_params(Box(
    drift_μ = (-0.5, 0.5),
    noise = (0, 2),
    threshold = (1, 10),
    sample_cost = (0, .02),
    between_σ = (0, 2),
    within_σ=0,
    judgement_noise=1,
));

optimal_tbl = fit_exp1_model("optimal", optimal_policies, optimal_prms)

# %% ==================== empirical ====================

print_header("empirical")

@everywhere begin
    @isdefined(emp_pretest_stop_dist) || const emp_pretest_stop_dist = empirical_distribution(@subset(pretest, :response_type .== "empty").rt)
    @isdefined(emp_crit_stop_dist) || const emp_crit_stop_dist = empirical_distribution(@subset(trials, :response_type .== "empty").rt)

    empirical_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
        RandomStoppingPolicy(exp1_mdp(prm), emp_crit_stop_dist),
    )
end

empirical_prms = sample_params(Box(
    drift_μ = (-0.5, 0.5),
    noise = (0, 2),
    threshold = (1, 10),
    sample_cost = 0,
    between_σ = (0, 2),
    within_σ=0,
    judgement_noise=1,
));

empirical_tbl = fit_exp1_model("empirical", empirical_policies, empirical_prms)
# %% --------


# # %% ==================== decision bound ====================

# print_header("decision bound")

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


# # %% --------

# opt_tbl = deserialize("tmp/exp1_fits_optimal")
# # opt_tbl.loss[1]

# # bound_tbl = deserialize("tmp/exp1_fits_bound")
# # bound_tbl.loss[1]


