@everywhere include("common.jl")
@everywhere include("exp2_simulate.jl")
mkpath("results/exp2")
mkpath("tmp")
Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
# %% --------

pretest = CSV.read("../data/processed/exp2/pretest.csv", DataFrame, missingstring="NA")
trials = CSV.read("../data/processed/exp2/trials.csv", DataFrame, missingstring="NA")
fixations = CSV.read("../data/processed/exp2/fixations.csv", DataFrame, missingstring="NA")

pretest = @rsubset pretest :practice == false :block == 3
trials = @rsubset trials :n_pres > 0

@everywhere trials = $trials
@everywhere pretest = $pretest
@everywhere fixations = $fixations

# %% ==================== summary statistics ====================

# @everywhere z_score(x) = (x .- mean(x)) ./ std(x)
@everywhere function exp2_sumstats(trials, fixations)
    accuracy = mean(trials.response_type .== "correct")

    tri = @chain trials begin
        @rsubset :response_type == "correct" 
        groupby([:wid, :pretest_accuracy_first, :pretest_accuracy_second])
        @combine begin
            :rt_μ = mean(:rt)
            :rt_σ = std(:rt)
            :choose_first = mean(:choose_first)
            :prop_first = mean(:total_first ./ :total_second)
            :total_first = mean(:total_first)
            :total_second = mean(:total_second)
            :n = length(:rt)
        end
    end

    fix = @chain fixations begin
        @rsubset :response_type == "correct"
        @rtransform :final = :presentation == :n_pres
        @rtransform :presentation = min(:presentation, 10)
        groupby([:wid, :presentation, :pretest_accuracy_first, :pretest_accuracy_second, :final])
        @combine begin
            :duration_μ = mean(:duration) 
            :duration_σ = std(:duration) 
            :n = length(:duration)
        end
    end

    (;accuracy, tri, fix)
end
target = exp2_sumstats(trials, fixations)

# %% ==================== loss ====================

function pooled_mean_std(ns::AbstractVector{<:Integer},
                        μs::AbstractVector{<:Number},
                        σs::AbstractVector{<:Number})
    nsum = sum(ns)
    meanc = ns' * μs / nsum
    vs = replace!(σs .^ 2, NaN=>0)
    varc = sum((ns .- 1) .* vs + ns .* abs2.(μs .- meanc)) / (nsum - 1)
    return meanc, .√(varc)
end

function nonfinal_duration(metric)    
    X = @chain metric.fix begin
        @rsubset :presentation > 1 && !:final
        groupby(:wid)
        @transform @astable begin
            μ, σ = pooled_mean_std(:n, :duration_μ, :duration_σ)
            :z = (:duration_μ .- μ) / σ
        end
        @rsubset !isnan(:z)
        @rtransform @astable begin
            x = (:pretest_accuracy_first, :pretest_accuracy_second)
            f, n = iseven(:presentation) ? reverse(x) : x
            :fixated = f
            :nonfixated = n
            :relative = f - n
        end
        @bywrap [:relative] mean(:z, Weights(:n))
    end
end

_human_nonfinal = nonfinal_duration(target)
function fix_loss(pred)
    p = nonfinal_duration(pred)
    size(_human_nonfinal) == size(p) || return Inf
    sum(squared.(_human_nonfinal .- p))
end

# %% ==================== optimal ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp2_mdp(prm)),
)

opt_prms = sobol(10000, Box(
    drift_μ = (-1, 1),
    noise = (.5, 2.5),
    drift_σ = (1, 3),
    threshold = (5, 15),
    sample_cost = (0, .1),
    switch_cost = (0, .05),
    strength_drift_μ = 0.,
    strength_drift_σ = 0.,
    judgement_noise=1,
))

# %% --------
mkpath(".cache/exp2_opt_sumstats")
opt_sumstats = @showprogress pmap(opt_prms) do prm
    cache(".cache/exp2_opt_sumstats/$(stringify(prm))") do
        df = simulate_exp2(optimal_policies, prm)
        x = exp2_sumstats(make_trials(df), make_fixations(df))
        GC.gc()
        x
    end
end;
serialize("tmp/exp2_opt_sumstats", opt_sumstats)

# %% --------
opt_sumstats = deserialize("tmp/exp2_opt_sumstats");
opt_prm, tbl, loss = minimize_loss(fix_loss, opt_sumstats, opt_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ])))

# %% --------

@time df = simulate_exp2(optimal_policies, opt_prm)
exp2_sumstats(make_trials(df), make_fixations(df))

df |> make_trials |> CSV.write("results/exp2/optimal_trials.csv")
df |> make_fixations |> CSV.write("results/exp2/optimal_fixations.csv")

# %% ==================== empirical ====================

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

emp_prms = sobol(10000, Box(
    drift_μ = (-1, 1),
    noise = (.5, 2.5),
    drift_σ = (1, 3),
    threshold = (5, 15),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    sample_cost = 0.,
    switch_cost = 0.,
));

# %% --------
mkpath(".cache/exp2_emp_sumstats")
emp_sumstats = @showprogress pmap(emp_prms) do prm
    cache(".cache/exp2_emp_sumstats/$(stringify(prm))") do
        df = simulate_exp2(empirical_policies, prm)
        x = exp2_sumstats(make_trials(df))
        GC.gc()
        x
    end
end;
serialize("tmp/exp2_emp_sumstats", emp_sumstats)

# %% --------
emp_sumstats = deserialize("tmp/exp2_emp_sumstats")
# %% --------
function full_loss(pred)
    pred.accuracy > .8 || return Inf
    fix_loss(pred)
end
emp_prm, tbl, loss = minimize_loss(full_loss, emp_sumstats, emp_prms);
display(select(tbl, Not([:strength_drift_μ, :strength_drift_σ])))

# %% --------
df = simulate_exp2(empirical_policies, emp_prm)
emp_ss = exp2_sumstats(make_trials(df), make_fixations(df))

df |> make_trials |> CSV.write("results/exp2/empirical_trials.csv")
df |> make_fixations |> CSV.write("results/exp2/empirical_fixations.csv")

# %% ==================== empirical ====================





# %% ==================== random ====================

@everywhere function random_policies(prm)
    pretest_stop_dist = Gamma(prm.α_pre, prm.θ_pre)
    crit_stop_dist = Gamma(prm.α_crit, prm.θ_crit)
    switch_dist = Gamma(prm.α_switch, prm.θ_switch)
    (
        RandomStoppingPolicy(pretest_mdp(prm), pretest_stop_dist),
        RandomSwitchingPolicy(exp1_mdp(prm), switch_dist, crit_stop_dist),
    )
end

rand_prms = sobol(10000, Box(
    α_pre = (1, 20),
    θ_pre = (1, 20),
    α_crit = (1, 20),
    θ_crit = (1, 20),
    α_fix = (1, 10),
    θ_fix = (1, 10),
    drift_μ = (-1, 1),
    noise = (.5, 2.5),
    drift_σ = (1, 3),
    threshold = (5, 15),
    strength_drift_μ = 0,
    strength_drift_σ = 0.,
    sample_cost = 0.,
));


