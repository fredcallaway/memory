if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end
# %% --------
@everywhere include("common.jl")
@everywhere include("exp1_base.jl")
mkpath("results/exp1")
mkpath("tmp")

N_SOBOL = 50_000

# %% --------  

function compute_sumstats(name, make_policies, prms; N=100000, read_only = false)
    dir = "cache/exp1_$(name)_sumstats_$(MS_PER_SAMPLE)_$N"
    mkpath(dir)
    map = read_only ? asyncmap : pmap
    @showprogress "sumstats" map(prms) do prm
        cache("$dir/$(hash(prm))"; read_only) do
            nprocs() == 1 && (print("."); flush(stdout))
            try
                exp1_sumstats(simulate_exp1(make_policies, prm, N))
            catch
                missing
            end
        end
    end;
end

# %% ==================== load data ====================

pretest = load_data("exp1/pretest")
trials = load_data("exp1/trials")
target = exp1_sumstats(trials);

@everywhere trials = $trials
@everywhere pretest = $pretest
@everywhere target = $target

# %% ==================== likelihood ====================
@everywhere using Optim

@everywhere function smooth_rt!(result, p::KeyedArray, d::Distribution, ε::Float64=1e-6)
    pd = diff([0; cdf(d, p.rt)])
    for h in axes(p, 4), i in axes(p, 3), j in axes(p, 2), z in axes(p, 1)
        result[z, j, i, h] = sum(1:z) do k
            y = z - k
            @inbounds p[k, j, i, h] * pd[y + 1]
        end
    end
    smooth_uniform!(result, ε)
end

@everywhere function optimize_rt_noise(ss)
    X = zeros(size(target.hist))
    optimize([10., 10.]) do x
        any(xi < 0 for xi in x) && return Inf
        smooth_rt!(X, ss.hist, Gamma(x...))
        crossentropy(target.hist, X)
    end
end

@everywhere function acc_rate(ss)
    x = ssum(ss.hist, :rt, :judgement)
    x ./= sum(x, dims=:response_type)
    x("correct")
end

@everywhere function loss(ss)
    ismissing(ss) && return Inf
    x = acc_rate(ss)
    (x[1] < .1 && x[3] > .85) || return Inf
    res = optimize_rt_noise(ss)
    res.minimum
end

function fit_exp1_model(name, make_policies, prms; n_top=1000, n_sim_top=1_000_000)
    sumstats = compute_sumstats(name, make_policies, prms);
    tbl = compute_loss(loss, sumstats, prms);
    serialize("tmp/exp1_tbl_$name", tbl)
    display(select(tbl, Not([:judgement_noise]))[1:13, :])

    top_prms = map(NamedTuple, eachrow(tbl[1:n_top, :]));
    top_sumstats = compute_sumstats(name, make_policies, top_prms; N=n_sim_top);
    top_tbl = compute_loss(loss, top_sumstats, top_prms)
    display(select(top_tbl, Not([:judgement_noise]))[1:13, :])

    rt_noise = pmap(eachrow(top_tbl)) do row
        Gamma(optimize_rt_noise(row.ss).minimizer...)
    end
    top_tbl.rt_α = getfield.(rt_noise, :α)
    top_tbl.rt_θ = getfield.(rt_noise, :θ)

    mkpath("results/exp1/$(name)_trials/")
    @showprogress "simulating" pmap(enumerate(eachrow(top_tbl)[1:10])) do (i, row)
        rt_noise = Gamma(row.rt_α, row.rt_θ)
        prm = NamedTuple(row)
        sim = simulate_exp1(make_policies, prm, 1_000_000)
        sim.rt = sim.rt .+ rand(rt_noise, nrow(sim))
        CSV.write("results/exp1/$(name)_trials/$i.csv", sim)
    end

    serialize("tmp/exp1_fits_$name", top_tbl)
    top_tbl
end

# %% ==================== optimal ====================

print_header("optimal")

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp1_mdp(prm)),
)

optimal_prms = sobol(N_SOBOL, Box(
    drift_μ = (-0.5, 0.5),
    noise = (0.1, 2.),
    drift_σ = (0.5, 2),
    threshold = (2, 20),
    sample_cost = (.001, .02),
    # strength_drift_μ = (-0.3, 0),
    # strength_drift_σ = (0, 0.3),
    strength_drift_μ = 0.,
    strength_drift_σ = 0.,
    judgement_noise=1,
));

optimal_tbl = fit_exp1_model("optimal", optimal_policies, optimal_prms)

# # %% ==================== empirical ====================

print_header("empirical")

@everywhere begin
    @isdefined(emp_pretest_stop_dist) || const emp_pretest_stop_dist = empirical_distribution(@subset(pretest, :response_type .== "empty").rt)
    @isdefined(emp_crit_stop_dist) || const emp_crit_stop_dist = empirical_distribution(@subset(trials, :response_type .== "empty").rt)

    empirical_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
        RandomStoppingPolicy(exp1_mdp(prm), emp_crit_stop_dist),
    )
end

empirical_prms = sobol(N_SOBOL, Box(
    drift_μ = (-0.5, 0.5),
    noise = (0.1, 2.),
    drift_σ = (0.5, 2),
    threshold = (2, 20),
    sample_cost = 0.,
    # strength_drift_μ = (-0.3, 0),
    # strength_drift_σ = (0, 0.3),
    strength_drift_μ = 0.,
    strength_drift_σ = 0.,
    judgement_noise=1,
));

empirical_tbl = fit_exp1_model("optimal", optimal_policies, empirical_prms)


# %% ==================== decision bound ====================

println("--- decision bound ---")

@everywhere bound_policies(prm) = (
    ConstantBoundPolicy(pretest_mdp(prm), prm.θ),
    ConstantBoundPolicy(exp1_mdp(prm), prm.θ),
)

bound_prms = sobol(N_SOBOL, Box(
    drift_μ = (-0.5, 0.5),
    noise = (0.1, 2.),
    drift_σ = (0.5, 2),
    threshold = (2, 20),
    sample_cost = 0.,
    θ = (1, 15),
    τ = (.001, 1, :log),
    # strength_drift_μ = (-0.3, 0),
    # strength_drift_σ = (0, 0.3),
    strength_drift_μ = 0.,
    strength_drift_σ = 0.,
    judgement_noise=1,
))

bound_sumstats = compute_sumstats("bound", bound_policies, bound_prms);

println("Done!")

