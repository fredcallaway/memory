if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end

@everywhere include("common.jl")
@everywhere include("exp2_base.jl")
mkpath("results/exp2")

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
    dir = "cache/exp2_$(name)_sumstats"
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

# %% ==================== optimal ====================

@everywhere optimal_policies(prm) = (
    OptimalPolicy(pretest_mdp(prm)),
    OptimalPolicy(exp2_mdp(prm)),
)

name = "optimal"
n_top = 10
top_table = deserialize("tmp/exp1_fits_$name")
exp1_top = eachrow(top_table[1:n_top, 1:8])

prms = map(product(exp1_top, 0:.001:.01)) do (prm, switch_cost)
    (;prm..., switch_cost)
end

sumstats = compute_sumstats("optimal", optimal_policies, prms)
target_fix = @with ss_human.tri mean(:n_pres, Weights(:n))

# %% --------
nfix_hist(ss) = @chain ss.tri begin
    @rtransform :n_pres = min(7, :n_pres)
    @rsubset !ismissing(:choose_first)
    wrap_pivot(:n, sum, n_pres=1:7)
    normalize!
end

L = map(sumstats) do ss
    crossentropy(nfix_hist(ss_human), smooth_uniform!(nfix_hist(ss), .01))
end

argmin(L; dims=2)

# %% --------

mkpath("results/exp2/optimal_trials/")
mkpath("results/exp2/optimal_fixations/")

function do_sims(prms)
    @showprogress "simulate" pmap(enumerate(prms)) do (i, prm)
        # _, j = findmin(minimum, dur_opts[i, :])
        # dur_noise = Gamma(dur_opts[i, j].minimizer...)
    
        sim = simulate_exp2(optimal_policies, prm)
        
        res = optimize_duration_noise(sim, human_fixations) # FIXME should be done on separate simulation?
        dur_noise = Gamma(res.minimizer...)
        add_duration_noise!(sim, dur_noise)

        trials = make_trials(sim); fixations = make_fixations(sim)
        CSV.write("results/exp2/optimal_trials/$i.csv", trials)
        CSV.write("results/exp2/optimal_fixations/$i.csv", fixations)
        exp2_sumstats(trials, fixations)
    end
end

do_sims(prms[argmin(L; dims=2)]);  # TODO: plot the results



# %% --------


prm = (;exp1_top[1]..., switch_cost = 0.005)
m = exp2_mdp(prm, maxt=15000)
B = BackwardsInduction(m; verbose=true)

pre_pol = OptimalPolicy(pretest_mdp(prm))
crit_pol = OptimalPolicy(B)

# %% --------

sim = simulate_exp2(pre_pol, crit_pol, 100_000)
trials = make_trials(sim); fixations = make_fixations(sim);
ss = exp2_sumstats(trials, fixations)

res = optimize_duration_noise(sim, human_fixations) # FIXME should be done on separate simulation?
dur_noise = Gamma(res.minimizer...)
add_duration_noise!(sim, dur_noise)

trials = make_trials(sim); fixations = make_fixations(sim)
CSV.write("results/exp2/optimal_trials.csv", trials)
CSV.write("results/exp2/optimal_fixations.csv", fixations)


# %% ==================== Fit pretest ====================

name = "optimal"
n_top = 2
top_table = deserialize("tmp/exp1_fits_$name")
exp1_top = eachrow(top_table[1:n_top, 1:8])
prms = map(product(exp1_top, 0:.1:1, -1:.1:0)) do (prm, μ_shift, σ_shift)
    drift_μ = prm.drift_μ + μ_shift
    drift_σ = prm.drift_σ + σ_shift
    (;prm..., drift_μ, drift_σ)
end

model_acc = @showprogress pmap(prms) do prm
    pol = OptimalPolicy(pretest_mdp(prm))
    normalize(counts(Int.(1 .+ 2 .* last.(sample_strengths(pol)))))
end

target_acc = @chain human_pretest begin
    groupby([:wid, :word])
    @combine :acc = mean(:response_type .== "correct")
    @bywrap :acc length(:acc)
    normalize
end

L_acc = map(model_acc) do p
    length(p) == length(target_acc) || return Inf
    crossentropy(target_acc, p)
end

# best_idx = argmin(L; dims=(2,3))
# updated_prms = prms[best_idx][:]

# %% --------

@everywhere function simulate_pretest(pol, N=1_000_000)
    map(1:N) do i
        sim = simulate(pol, fix_log=RTLog())
        (response_type = sim.b.focused == -1 ? "empty" : "correct",
         rt=sim.fix_log.rt * MS_PER_SAMPLE)
    end |> DataFrame
end

@everywhere function pretest_hist(pretest::DataFrame; dt=MS_PER_SAMPLE, maxt=MAX_TIME)
    @chain pretest begin
        @rtransform :rt = quantize(:rt, dt)
        wrap_counts(
            rt=dt:dt:maxt, 
            response_type=["correct", "empty"], 
        ) 
        normalize!
    end
end

model_rt = @showprogress pmap(prms) do prm
    pol = OptimalPolicy(pretest_mdp(prm))
    pretest_hist(simulate_pretest(pol))
end

target_rt = @chain human_pretest begin
    @rsubset :response_type == "correct"
    @rsubset :rt ≤ MAX_TIME
    pretest_hist
    _("correct")
end

L_rt = @showprogress pmap(model_rt) do x
    res = optimize_duration_noise(normalize(x("correct")), target_rt; key=:rt)
    res.minimum
end

idx_acc = argmin(L_acc; dims=(2,3))
idx_rt = argmin(L_rt; dims=(2,3))
idx_both = argmin(L_acc .+ L_rt; dims=(2,3))


# %% ==================== tune to experiment 2 ====================

function compute_sumstats(name, make_policies, prms; read_only = false)
    dir = "cache/exp2_$(name)_sumstats"
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

updated_prms = prms[[idx_acc; idx_both; idx_rt]][:]

updated_prms = map(product(updated_prms, 0:.0025:.02)) do (prm, switch_cost)
    (;prm..., switch_cost)
end

sumstats = compute_sumstats("optimal", optimal_policies, updated_prms);

# %% ==================== loss ====================

const ss_human = exp2_sumstats(human_trials, human_fixations);

nfix_hist(ss) = @chain ss.tri begin
    @rsubset :n_pres ≤ 7
    wrap_pivot(:n, sum, n_pres=1:7)
    normalize    
end

target = nfix_hist(ss_human)

L = map(sumstats) do ss
    pred = nfix_hist(ss)
    crossentropy(target, pred)
end

final_prms = updated_prms[argmin(L; dims=2)][:]

display(select(DataFrame(final_prms), Not([:judgement_noise])))

# %% ==================== simulate ====================

mkpath("results/exp2/optimal_trials/")
mkpath("results/exp2/optimal_fixations/")

function do_sims(prms)
    @showprogress "simulate" pmap(enumerate(prms)) do (i, prm)
        # _, j = findmin(minimum, dur_opts[i, :])
        # dur_noise = Gamma(dur_opts[i, j].minimizer...)
    
        sim = simulate_exp2(optimal_policies, prm)
        
        res = optimize_duration_noise(sim, human_fixations) # FIXME should be done on separate simulation?
        dur_noise = Gamma(res.minimizer...)
        add_duration_noise!(sim, dur_noise)

        trials = make_trials(sim); fixations = make_fixations(sim)
        CSV.write("results/exp2/optimal_trials/$i.csv", trials)
        CSV.write("results/exp2/optimal_fixations/$i.csv", fixations)
        exp2_sumstats(trials, fixations)
    end
end

final_sumstats = do_sims(final_prms)

ss = final_sumstats[1]

nfix_hist(final_sumstats[6])
nfix_hist(ss_human)




# %% ==================== fit joint ====================

prms = map(product(exp1_top, 0:.1:1, -1:.1:0, 0:.01:.01)) do (prm, μ_shift, σ_shift, switch_cost)
    drift_μ = prm.drift_μ + μ_shift
    drift_σ = prm.drift_σ + σ_shift
    (;prm..., drift_μ, drift_σ, switch_cost)
end

sumstats = compute_sumstats("optimal", optimal_policies, prms);

# %% --------

function make_hist(ss)
    @chain ss.tri begin
        @rtransform :n_pres = min(:n_pres, 7)
        @rtransform :outcome = (ismissing(:choose_first) ? "error" : :choose_first ? "first" : "second")
        wrap_pivot(:n, sum, 
            pretest_accuracy_first=0:0.5:1,
            pretest_accuracy_second=0:0.5:1,
            outcome=["error", "first", "second"],
            n_pres=1:7,
        )
        normalize!
    end
end

# %% --------
# const ss_human = exp2_sumstats(human_trials, human_fixations);
const human_hist = make_hist(ss_human);

@everywhere function loss(ss)
    # ("n_pres" in names(ss.tri)) || return Inf
    ismissing(ss) && return Inf
    crossentropy(human_hist, smooth_uniform!(make_hist(ss), .01))
end

L = map(loss, sumstats)
tbl = DataFrame(prms[argmin(L, dims=(2,3))][:])
display(select(tbl, Not([:judgement_noise, :strength_drift_μ, :strength_drift_σ])))

# %% --------
ss = sumstats[argmin(L)]

ss = sumstats[1, 2, 1]
wrap_pivot(ss.tri, :n, sum, n_pres=1:100)[1:10] / sum(ss.tri.n)

(@bywrap ss.tri :n_pres sum(:n)) |> normalize
(@bywrap ss.tri :choose_first sum(:n)) |> normalize


# %% --------

function pretest_accuracy_rate(ss)
    p = collect((@bywrap ss.tri :pretest_accuracy_first sum(:n))) + 
        collect((@bywrap ss.tri :pretest_accuracy_second sum(:n)))
    p = normalize(p)
end

function acc_loss(ss)
    crossentropy(pretest_accuracy_rate(ss_human), pretest_accuracy_rate(ss))
end

L = acc_loss.(sumstats)

pretest_accuracy_rate(sumstats[3,1,4])
pretest_accuracy_rate(ss_human)

prms[2,1,5]



# %% --------

function fix_sumstats(name, make_policies, prms)
    dir = "cache/exp2_$(name)_sumstats_fixed"
    mkpath(dir)
    @showprogress pmap(prms) do prm
        cache("$dir/$(hash(prm))") do
            ss = deserialize("cache/exp2_$(name)_sumstats/$(stringify(prm))")
            if !("n_pres" in names(ss.tri))
                sim = simulate_exp2(make_policies, prm)
                res = optimize_duration_noise(sim, human_fixations)
                add_duration_noise!(sim, Gamma(res.minimizer...))
                x = exp2_sumstats(make_trials(sim), make_fixations(sim))
                GC.gc()
                ss = (;x..., duration_loss=res.minimum)
            end
            ss
        end
    end
end

sumstats = fix_sumstats("optimal", optimal_policies, prms);


# %% --------

mkpath("cache/exp2_optimal_duropt")
dur_opts = @showprogress "mle " pmap(prms) do prm
    cache("cache/exp2_optimal_duropt/$(hash(prm))"; read_only=true) do
        sim = simulate_exp2(optimal_policies, prm)
    end
end

# %% --------

i = 1
_, j = findmin(minimum, dur_opts[i, :])
prm = prms[i, j]

m = exp2_mdp(prm; maxt=15000)
B = BackwardsInduction(m; verbose=true)

pre_pol = OptimalPolicy(pretest_mdp(prm))
crit_pol = OptimalPolicy(B)
sim = simulate_exp2(pre_pol, crit_pol)
add_duration_noise!(sim, Gamma(dur_opts[i, j].minimizer...))

trials = make_trials(sim); fixations = make_fixations(sim);
CSV.write("results/exp2/optimal_trials.csv", trials)
CSV.write("results/exp2/optimal_fixations.csv", fixations)

# %% --------
sim = simulate_exp2(pre_pol, crit_pol)
trials = make_trials(sim); fixations = make_fixations(sim);
ss1 = exp2_sumstats(trials, fixations)
normalize(counts(trials.n_pres))
mean(trials.response_type .== "correct")


# %% --------
mkpath("results/exp2/optimal_trials/")
mkpath("results/exp2/optimal_fixations/")

function do_sims(jobs, dur_opts)
    @showprogress "simulate" pmap(1:n_top) do i
        _, j = findmin(minimum, dur_opts[i, :])
        dur_noise = Gamma(dur_opts[i, j].minimizer...)
        
        sim = simulate_exp2(optimal_policies, prms[i, j])
        add_duration_noise!(sim, dur_noise)

        trials = make_trials(sim); fixations = make_fixations(sim)
        CSV.write("results/exp2/optimal_trials/$i.csv", trials)
        CSV.write("results/exp2/optimal_fixations/$i.csv", fixations)
        exp2_sumstats(trials, fixations)
    end
end

do_sims(jobs, dur_opts)


# # %% --------

# opt_prm = deserialize("tmp/exp1_opt_prm")
# opt_prm = (;opt_prm..., switch_cost=.01)
# @time opt_df = simulate_exp2(optimal_policies, prm, 1000000);
# opt_dur_noise = mle_duration_noise(opt_df, fixations)
# add_duration_noise!(opt_df, opt_dur_noise)

# opt_trials = make_trials(opt_df); opt_fixations = make_fixations(opt_df)
# CSV.write("results/exp2/optimal_trials.csv", opt_trials)
# CSV.write("results/exp2/optimal_fixations.csv", opt_fixations)

# # %% ==================== empirical ====================

# plausible_skips(x) = @rsubset(x, :response_type in ["other", "empty"])
# const emp_pretest_stop_dist = empirical_distribution(plausible_skips(human_pretest).rt)
# const emp_crit_stop_dist = empirical_distribution(skipmissing(plausible_skips(human_trials).rt))
# const emp_switch_dist = empirical_distribution(fixations.duration)

# empirical_policies(prm) = (
#     RandomStoppingPolicy(pretest_mdp(prm), emp_pretest_stop_dist),
#     RandomSwitchingPolicy(exp2_mdp(prm), emp_switch_dist, emp_crit_stop_dist),
# )

# emp_prm = deserialize("tmp/exp1_emp_emp_prm")
# emp_prm = (;emp_prm..., switch_cost=NaN)
# @time emp_df = simulate_exp2(empirical_policies, emp_prm, 1000000);
# emp_dur_noise = mle_duration_noise(emp_df, fixations)
# add_duration_noise!(emp_df, emp_dur_noise)

# emp_trials = make_trials(emp_df); emp_fixations = make_fixations(emp_df)
# CSV.write("results/exp2/empirical_trials.csv", emp_trials)
# CSV.write("results/exp2/empirical_fixations.csv", emp_fixations)
