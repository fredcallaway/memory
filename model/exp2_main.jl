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
@everywhere fixations = $fixations

# %% ==================== Define Metrics ====================

@everywhere z_score(x) = (x .- mean(x)) ./ std(x)
@everywhere function critical_metrics(trials)
    nfix = counts(trials.n_pres, 1:4) ./ nrow(trials)
    push!(nfix, mean(trials.n_pres .> 4))
    accuracy = mean(trials.response_type .== "correct")
    correct_trials = @chain trials begin
        @rsubset :response_type == "correct" 
        @transform :rel_pretest = :pretest_accuracy_first - :pretest_accuracy_second
    end

    choice_rate = @bywrap correct_trials :pretest_accuracy_first mean(:choose_first)
    rt_μ, rt_σ = @with correct_trials (mean(:rt), std(:rt))
    fix1 = @chain correct_trials begin
        @rsubset :n_pres > 1
        groupby(:wid)
        @transform :first_pres_time = z_score(:first_pres_time)
        @bywrap :pretest_accuracy_first nanmean(:first_pres_time)
    end

    fix2 = @chain correct_trials begin
        @rsubset :n_pres > 2
        groupby(:wid)
        @transform :second_pres_time = z_score(:second_pres_time)
        @bywrap :rel_pretest nanmean(:second_pres_time)
    end

    fix3 = @chain correct_trials begin
        @rsubset :n_pres > 3
        groupby(:wid)
        @transform :third_pres_time = z_score(:third_pres_time)
        @bywrap :rel_pretest nanmean(:third_pres_time)
    end

    fixs = [:first_pres_time, :second_pres_time, :third_pres_time, :last_pres_time]
    fix_times = map(fixs) do f
        mean(skipmissing(correct_trials[:, f]))
    end
    (;nfix, accuracy, choice_rate, rt_μ, rt_σ, fix1, fix2, fix3, fix_times)
end
target = critical_metrics(trials)

# %% ==================== Compute optimal metrics ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp2_mdp(prm)),
)

prms = sobol(10000, Box(
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

mkpath(".cache/exp2_crit_metrics")

crit_metrics = @showprogress asyncmap(prms) do prm
    cache(".cache/exp2_crit_metrics/$(stringify(prm))") do
        @assert false
        df = simulate_exp2(optimal_policies, prm)
        x = critical_metrics(make_trials(df))
        GC.gc()
        x
    end
end;
serialize("tmp/exp2_crit_metrics", crit_metrics)
# %% --------
crit_metrics = deserialize("tmp/exp2_crit_metrics");

# %% ==================== Loss ====================

function fix_loss(pred)
    mapreduce(+, [:fix1, :fix2, :fix3]) do f
        t = getfield(target, f)
        p = getfield(pred, f)
        size(t) == size(p) || return Inf
        sum(squared.(t .- p))
    end
end

function minimize_loss(loss, metrics, prms)
    L = map(loss, metrics);
    flat_prms = collect(prms)[:];
    flat_L = collect(L)[:];
    fit_prm = flat_prms[argmin(flat_L)]
    tbl = flat_prms[sortperm(flat_L)] |> DataFrame
    tbl.loss = sort(flat_L)
    fit_prm, tbl, flat_L
end

opt_prm, tbl, loss = minimize_loss(fix_loss, crit_metrics, prms);
display(select(tbl, Not(:strength_drift_μ)))

# %% --------
opt_prm

@time df = simulate_exp2(optimal_policies, opt_prm)
make_trials(df) |> critical_metrics |> fix_loss
df |> make_trials |> CSV.write("results/exp2/optimal_trials.csv")
df |> make_fixations |> CSV.write("results/exp2/optimal_fixations.csv")

# %% --------


