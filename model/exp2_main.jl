if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

@everywhere include("common.jl")
@everywhere include("exp2_base.jl")
mkpath("results/noise_exp2")

# %% ==================== load data ====================

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

function compute_sumstats(name, make_policies, prms; read_only = false)
    dir = "cache/noise_exp2_$(name)_sumstats"
    mkpath(dir)
    map = read_only ? asyncmap : pmap
    @showprogress map(prms) do prm
        cache("$dir/$(hash(prm))"; read_only) do
            sim = simulate_exp2(make_policies, prm)
            res = optimize_duration_noise(sim, human_fixations)
            add_duration_noise!(sim, Gamma(res.minimizer...))
            x = exp2_sumstats(make_trials(sim), make_fixations(sim))
            GC.gc()
            (;x..., duration_opt=res)
        end
    end;
end

function write_sims(name, make_policies; n_top=5)
    top_table = select(deserialize("tmp/noise_exp1_fits_$name"), Not(:ss))
    exp1_top = eachrow(top_table)[1:n_top]

    prms = map(exp1_top) do prm
        (;prm..., switch_cost=prm.sample_cost)
    end

    mkpath("results/noise_exp2/$(name)_trials/")
    mkpath("results/noise_exp2/$(name)_fixations/")

    @showprogress "simulate" pmap(enumerate(prms)) do (i, prm)
        pre_pol, crit_pol = make_policies(prm)
        sim = simulate_exp2(pre_pol, crit_pol)
        # res = optimize_duration_noise(sim, human_fixations)
        # dur_noise = Gamma(res.minimizer...)
        dur_noise = Gamma(prm.rt_α, prm.rt_θ)

        sim = simulate_exp2(pre_pol, crit_pol)
        add_duration_noise!(sim, dur_noise)

        trials = make_trials(sim); fixations = make_fixations(sim)
        CSV.write("results/noise_exp2/$(name)_trials/$i.csv", trials)
        CSV.write("results/noise_exp2/$(name)_fixations/$i.csv", fixations)
        ss = exp2_sumstats(trials, fixations)
        (;ss..., res)
    end
end

# %% ==================== optimal ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp2_mdp(prm)),
)

optimal_results = write_sims("optimal", optimal_policies)
serialize("tmp/noise_exp2_optimal_results", optimal_results)

# # %% ==================== empirical ====================

# @everywhere begin
#     plausible_skips(x) = @rsubset(x, :response_type in ["other", "empty"])
#     const emp_pretest_stop_dist = empirical_distribution(plausible_skips(human_pretest).rt)
#     const emp_crit_stop_dist = empirical_distribution(skipmissing(plausible_skips(human_trials).rt))
#     const emp_switch_dist = empirical_distribution(human_fixations.duration)

#     empirical_policies(prm) = (
#         RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
#         RandomSwitchingPolicy(exp2_mdp(prm), emp_switch_dist, emp_crit_stop_dist),
#     )
# end
# empirical_results = write_sims("empirical", empirical_policies)

# mean(Gamma(empirical_results[1].res.minimizer...))


