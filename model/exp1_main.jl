if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end
# %% --------
@everywhere include("common.jl")
@everywhere include("exp1_base.jl")
mkpath("results/exp1")
N_SOBOL = 10_000

if isempty(ARGS) 
    CACHE_ONLY = false
    JOBS = 1:N_SOBOL
else  # precomputing with batch compute
    CACHE_ONLY = true
    JOBS = let
        chunk_size = 100
        j = parse(Int, ARGS[1])
        range(1 + chunk_size * (j-1), length=chunk_size)
    end
end

# %% --------  

function compute_sumstats(name, make_policies, prms; read_only = nprocs() == 1)
    mkpath("cache/exp1_$(name)_sumstats_50")
    map_ = read_only ? asyncmap : nprocs() > 1 ? pmap : map
    map_(prms) do prm
        # nprocs() == 1 && (print("."); flush(stdout))
        cache("cache/exp1_$(name)_sumstats_50/$(hash(prm))"; read_only) do
            try
                exp1_sumstats(simulate_exp1(make_policies, prm))
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
    result .*= (1 - ε * length(result))
    result .+= ε
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

# %% ==================== optimal ====================

println("--- optimal ---")

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp1_mdp(prm)),
)

opt_prms = sobol(N_SOBOL, Box(
    drift_μ = (-0.5, 0.5),
    noise = (0.1, 1.5),
    drift_σ = (0.1, 2),
    threshold = (2, 15),
    sample_cost = (.005, .02),
    strength_drift_μ = (-0.3, 0),
    strength_drift_σ = (0, 0.3),
    judgement_noise=1,
));

opt_sumstats = compute_sumstats("opt", optimal_policies, opt_prms[JOBS]);

# %% --------
if !CACHE_ONLY
    opt_prm, opt_ss, tbl, full_loss = minimize_loss(loss, opt_sumstats, opt_prms) 
    display(select(tbl, Not([:judgement_noise]))[1:13, :])
    df = simulate_exp1(optimal_policies, opt_prm)
    rt_noise = Gamma(optimize_rt_noise(opt_ss).minimizer...)
    df.rt = df.rt .+ rand(rt_noise, nrow(df))
    CSV.write("results/exp1/optimal_trials.csv", df)
end

# # %% ==================== empirical ====================
println("--- empirical ---")

# %% --------

@everywhere begin
    @isdefined(emp_pretest_stop_dist) || const emp_pretest_stop_dist = empirical_distribution(@subset(pretest, :response_type .== "empty").rt)
    @isdefined(emp_crit_stop_dist) || const emp_crit_stop_dist = empirical_distribution(@subset(trials, :response_type .== "empty").rt)

    empirical_policies(prm) = (
        RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
        RandomStoppingPolicy(exp1_mdp(prm), emp_crit_stop_dist),
    )
end

emp_prms = sobol(N_SOBOL, Box(
    drift_μ = (-0.5, 0.5),
    noise = (0.5, 2.),
    drift_σ = (1, 2),
    threshold = (5, 15),
    sample_cost = 0.,
    strength_drift_μ = (-0.3, 0),
    strength_drift_σ = (0, 0.3),
    judgement_noise=1,
));

emp_sumstats = compute_sumstats("emp", empirical_policies, emp_prms[JOBS]);

# # %% --------

if !CACHE_ONLY
    emp_prm, emp_ss, tbl, full_loss = minimize_loss(loss, emp_sumstats, emp_prms);
    display(select(tbl, Not([:judgement_noise, :sample_cost]))[1:13, :])

    df = simulate_exp1(empirical_policies, emp_prm)
    rt_noise = Gamma(optimize_rt_noise(emp_ss).minimizer...)
    df.rt = df.rt .+ rand(rt_noise, nrow(df))
    CSV.write("results/exp1/empirical_trials.csv", df)
end

# %% ==================== decision bound ====================

println("--- decision bound ---")

@everywhere bound_policies(prm) = (
    ConstantBoundPolicy(pretest_mdp(prm), prm.θ),
    ConstantBoundPolicy(exp1_mdp(prm), prm.θ),
)

bound_prms = sobol(N_SOBOL, Box(
    drift_μ = (-1, 1),
    noise = (1., 3.),
    drift_σ = (0.5, 2.5),
    threshold = (5, 15),
    θ = (1, 15),
    τ = (.001, 1, :log),
    strength_drift_μ = (-0.3, 0),
    strength_drift_σ = (0, 0.3),
    judgement_noise=1,
    sample_cost = 0.,
))

bound_sumstats = compute_sumstats("bound", bound_policies, bound_prms);

# # %% --------

# bound_prm, bound_ss, tbl, full_loss = minimize_loss(loss, bound_sumstats, bound_prms);
# display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ, :judgement_noise, :sample_cost]))[1:13, :])
# df = simulate_exp1(bound_policies, bound_prm)
# @show loss(exp1_sumstats(df))
# CSV.write("results/exp1/bound_trials.csv", df)


println("Done!")

