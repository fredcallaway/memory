
@everywhere include("common.jl")
@everywhere include("exp1_base.jl")
mkpath("results/exp1")
mkpath("tmp")

N_SOBOL = 10_000
RUN = "apr18"

if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

# %% ==================== load data ====================

full_pretest = load_data("exp1/pretest")
full_trials = load_data("exp1/trials")
full_target = exp1_sumstats(trials);

# trials and target are used in exp1_base.jl... terrible i know
@everywhere trials = $full_trials
@everywhere pretest = $full_pretest
@everywhere target = $full_target

# %% ==================== experiment 1 ====================

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

opt_sumstats = compute_sumstats("optimal", optimal_policies, optimal_prms, read_only=true);

function sample_top()
    grp = collect(groupby(full_trials, :wid));
    trials = reduce(vcat, sample(grp, length(grp)))
    target = exp1_sumstats(trials);
    @everywhere trials = $trials
    @everywhere target = $target
    first(eachrow(compute_loss(loss, deepcopy(opt_sumstats), optimal_prms)))
end
possible_exp1_top = [sample_top() for i in 1:30]

# %% --------
mkpath("results/$(RUN)_robustness_exp1/optimal_trials")
exp1_results = @showprogress pmap(enumerate(possible_exp1_top)) do (i, row)
    rt_noise = Gamma(optimize_rt_noise(row.ss).minimizer...)
    prm = NamedTuple(row)
    sim = simulate_exp1(optimal_policies, prm, 1_000_000)
    sim.rt = sim.rt .+ rand(rt_noise, nrow(sim))
    CSV.write("results/$(RUN)_robustness_exp1/optimal_trials/$i.csv", sim)
    sim.rt = min.(sim.rt, 15000)
    exp1_sumstats(sim)
end

# %% --------

acc_rt(ss) = @bywrap ss.rt [:pretest_accuracy, :response_type] (sum(:n) > 10 ? mean(:μ, Weights(:n)) : missing)
judge_rt(ss) = @bywrap ss.rt [:judgement, :response_type] (sum(:n) > 10 ? mean(:μ, Weights(:n)) : missing)

function report(f, name, results)
    res = map(f, results)
    println("\n", name)
    for x in invert(res)
        μ, σ, mn, mx = map(Int ∘ round, juxt(mean, std, minimum, maximum)(x))
        println("   $μ ± $σ [$mn, $mx]")
    end
        # @info name mean(x) std(x) minimum(x) maximum(x)
end

report("correct rt", exp1_results) do ss
    x = acc_rt(ss)("correct")
    [x(1), x(0.5), x(0.5) - x(1)]
end

report("empty rt", exp1_results) do ss
    x = acc_rt(ss)("empty")
    [x(0), x(0.5), x(1), x(1) - x(0), x(1) - x(0.5)]
end

report("correct rt judgement", exp1_results) do ss
    judge_rt(ss)("correct")
end

report("correct rt judgement deltas", exp1_results) do ss
    x = judge_rt(ss)("correct")
    diff(collect(x))
end

report("empty rt judgement", exp1_results) do ss
    judge_rt(ss)("empty")
end

report("empty rt judgement deltas", exp1_results) do ss
    x = judge_rt(ss)("empty")
    diff(collect(x))
end

# %% ==================== experiment 2 ====================

@everywhere include("exp2_base.jl")

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp2_mdp(prm)),
)

human_pretest = CSV.read("../data/processed/exp2/pretest.csv", DataFrame, missingstring="NA")
human_trials = CSV.read("../data/processed/exp2/trials.csv", DataFrame, missingstring="NA")
human_fixations = CSV.read("../data/processed/exp2/fixations.csv", DataFrame, missingstring="NA")

human_pretest = @rsubset human_pretest :practice == false :block == 3
human_trials = @chain human_trials begin
    @rsubset :n_pres > 0
    @rsubset :response_type != "intrusion"
    @rtransform :choose_first = :response_type == "correct" ? :choose_first : missing
end

@everywhere human_trials = $human_trials
@everywhere human_pretest = $human_pretest
@everywhere human_fixations = $human_fixations

const ss_human = exp2_sumstats(human_trials, human_fixations);


prms = map(possible_exp1_top) do prm
    (;prm..., switch_cost=prm.sample_cost)
end

mkpath("results/$(RUN)_robustness_exp2/optimal_trials/")
mkpath("results/$(RUN)_robustness_exp2/optimal_fixations/")

exp2_results = @showprogress "simulate" pmap(enumerate(prms)) do (i, prm)
    pre_pol, crit_pol = optimal_policies(prm)
    sim = simulate_exp2(pre_pol, crit_pol; prm.within_σ, prm.between_σ)
    res = optimize_duration_noise(sim, human_fixations)
    dur_noise = Gamma(res.minimizer...)

    sim = simulate_exp2(pre_pol, crit_pol; prm.within_σ, prm.between_σ)
    add_duration_noise!(sim, dur_noise)

    trials = make_trials(sim); fixations = make_fixations(sim)
    CSV.write("results/$(RUN)_robustness_exp2/optimal_trials/$i.csv", trials)
    CSV.write("results/$(RUN)_robustness_exp2/optimal_fixations/$i.csv", fixations)
    ss = exp2_sumstats(trials, fixations)
    ss
    # (;ss..., res)
end

# %% --------

function fixated_duration(ss)
    @chain ss.fix begin
        @rsubset !:final
        @rtransform :fixated = isodd(:presentation) ? :pretest_accuracy_first : :pretest_accuracy_second
        @by [:wid, :fixated] :duration = mean(:duration_μ, Weights(:n))
        @bywrap [:fixated] mean(:duration)
    end
end

function nonfixated_duration(ss)
    @chain ss.fix begin
        @rsubset !:final && :presentation > 1
        @rtransform :nonfixated = iseven(:presentation) ? :pretest_accuracy_first : :pretest_accuracy_second
        @by [:wid, :nonfixated] :duration = mean(:duration_μ, Weights(:n))
        @bywrap [:nonfixated] mean(:duration)
    end
end

function final_nonfinal(ss)
    @chain ss.fix begin
        # @rsubset !:final && :presentation > 1
        # @rtransform :nonfixated = iseven(:presentation) ? :pretest_accuracy_first : :pretest_accuracy_second
        @by [:wid, :final] :duration = mean(:duration_μ, Weights(:n))
        @bywrap [:final] mean(:duration)
    end
end

report("fixated durations", exp2_results) do ss
    fixated_duration(ss)
end

report("fixated durations delta", exp2_results) do ss
    diff(collect(fixated_duration(ss)))
end

report("nonfixated durations", exp2_results) do ss
    nonfixated_duration(ss)
end

report("nonfixated durations delta", exp2_results) do ss
    diff(collect(nonfixated_duration(ss)))
end

report("final_nonfinal", exp2_results) do ss
    final_nonfinal(ss)
end

report("final_nonfinal diff", exp2_results) do ss
    x = final_nonfinal(ss)
    diff(collect(x))
end
