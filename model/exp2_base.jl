# %% ==================== simulate ====================

function exp2_mdp(prm; maxt=MAX_TIME)
    time_cost = @isdefined(PRETEST_COST) ? (MS_PER_SAMPLE / MAX_TIME) * .25 : 0
    m2 = MetaMDP{2}(;allow_stop=true, max_step=Int(maxt / MS_PER_SAMPLE), miss_cost=2,
        prm.threshold, prm.noise,
        sample_cost=prm.sample_cost + time_cost, prm.switch_cost,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end

function simulate_exp2(make_policies::Function, prm::NamedTuple, N=1_000_000)
    simulate_exp2(make_policies(prm)..., N; prm.within_σ, prm.between_σ)
end

function simulate_exp2(pre_pol::Policy, crit_pol::Policy, N=1_000_000; within_σ, between_σ)
    strengths = sample_strengths(pre_pol,  2N; within_σ, between_σ)
    pairs = map(1:2:2N) do i
        s1, pretest_accuracy_first = strengths[i]
        s2, pretest_accuracy_second = strengths[i+1]
        (;pretest_accuracy_first, pretest_accuracy_second), (s1, s2)
    end

    map(pairs) do (pretest_accuracy, s)
        sim = simulate(crit_pol; s, fix_log=FullFixLog())
        presentation_times = sim.fix_log.fixations .* float(MS_PER_SAMPLE)
        # presentation_times .+= (crit_pol.m.switch_cost / crit_pol.m.sample_cost) * MS_PER_SAMPLE
        (;response_type = sim.b.focused == -1 ? "empty" : "correct",
          choose_first = sim.b.focused == -1 ? missing : sim.b.focused == 1,
          pretest_accuracy..., 
          strength_first = s[1], strength_second = s[2],
          presentation_times
         )
    end |> DataFrame
end

function make_trials(df)
    safeindex(x, i) = length(x) < i ? NaN : x[i]
    @chain df begin
         @rtransform begin
            :first_pres_time = safeindex(:presentation_times, 1)
            :second_pres_time = safeindex(:presentation_times, 2)
            :third_pres_time = safeindex(:presentation_times, 3)
            :last_pres_time = :presentation_times[end]
            :n_pres = length(:presentation_times)
            :total_first = sum(:presentation_times[1:2:end])
            :total_second = sum(:presentation_times[2:2:end])
            :wid = "optimal"
        end
        @transform begin
            :rt = :total_first .+ :total_second
            :trial_id = 1:nrow(df)
        end
        select(names(human_trials))
    end
end

function make_fixations(df)
    @chain df begin
        @transform(:trial_id = 1:nrow(df))
        @rtransform(:n_pres = length(:presentation_times))
        @rtransform(:presentation = 1:(:n_pres), :wid = "optimal")
        DataFrames.flatten([:presentation_times, :presentation])
        DataFrames.rename(:presentation_times => :duration)
        select(names(human_fixations))
    end
end

# %% ==================== likelihood ====================

# this was used for flexible (not flexible2)
# function make_hist(trials::DataFrame; dt=MS_PER_SAMPLE, maxt=MAX_TIME)
#     @chain trials begin
#         @rsubset :response_type == "correct"
#         # @rsubset :response_type in ("correct", "empty")
#         @rtransform! :response = (
#             :response_type == "empty" ? "skip" :
#             :choose_first ? "first" :
#             "second"
#         )
#         @rtransform! :n_pres = clip(:n_pres, 0, 5)
#         @rtransform! :rt = quantize(:rt, dt)
#         wrap_counts(
#             rt=dt:dt:maxt, 
#             response=["first", "second"], 
#             # response=["skip", "first", "second"], 
#             n_pres=1:5,
#             pretest_accuracy_first=0:0.5:1,
#             pretest_accuracy_second=0:0.5:1,
#         ) 
#         normalize!
#     end
# end

# function make_hist(trials::DataFrame; dt=MS_PER_SAMPLE, maxt=MAX_TIME)
#     @chain trials begin
#         @rsubset :response_type == "correct"
#         # @rsubset :response_type in ("correct", "empty")
#         @rtransform! :n_pres = clip(:n_pres, 0, 5)
#         @rtransform! :rt = quantize(:rt, dt)
#         wrap_counts(
#             rt=dt:dt:maxt, 
#             n_pres=1:5,
#         ) 
#         normalize!
#     end
# end


# max_fix_time = @chain human_fixations begin
#     @rsubset :presentation == :n_pres
#     @with quantile(:duration, 0.95)
#     quantize(100)
# end  -> 4600 (hardcoding this)

function make_hist(fixations::DataFrame; dt=MS_PER_SAMPLE, maxt=4600)
    @chain fixations begin
        @rsubset :response_type == "correct"
        @rtransform! begin 
            :fixated = isodd(:presentation) ? :pretest_accuracy_first : :pretest_accuracy_second
            :nonfixated = iseven(:presentation) ? :pretest_accuracy_first : :pretest_accuracy_second
            :duration = clip(quantize(:duration, dt), 0, maxt)
            :final = :presentation == :n_pres
            :presentation = clip(:presentation, 0, 4)
        end
        wrap_counts(
            duration=dt:dt:maxt, 
            # response=["first", "second"], 
            final=[false,true],
            presentation=1:4,
            fixated=0:0.5:1,
            nonfixated=0:0.5:1,
        ) 
        normalize!
    end
end

function compute_histograms(name, make_policies, prms; N=100000, read_only=false, enable_cache=true)
    compute_cached("$(name)_histograms_$N", prms) do prm
        make_hist(make_fixations(simulate_exp2(make_policies, prm, N)))
    end
end

function compute_loss(histograms, prms; sort=true)
    tbl = DataFrame(prms)
    results = @showprogress "loss " pmap(histograms, prms) do model_hist, prm
        ismissing(model_hist) && return [Inf, 1000., 1000.]  # 1000 gives super long RT (a warning flag)
        human_hist2 = ssum(human_hist, :presentation)
        model_hist2 = ssum(model_hist, :presentation)
        if hasfield(typeof(prm), :α_ndt)
            (;α_ndt, θ_ndt) = prm
            lk = likelihood(model_hist2, human_hist2, Gamma(α_ndt, θ_ndt))
            [lk, α_ndt, θ_ndt]
        else
            res = optimize_ndt(model_hist2, human_hist2)
            [res.minimum; res.minimizer]
        end
    end
    tbl.loss, tbl.α_ndt, tbl.θ_ndt = invert(results)
    sort && sort!(tbl, :loss)
    tbl
end


# %% ==================== NDT on fixation durations ====================

function duration_hist(fixations; nonfinal=false, dt=MS_PER_SAMPLE, maxt=MAX_TIME)
    @chain fixations begin
        @rsubset :duration < MAX_TIME && (!nonfinal || :presentation ≠ :n_pres)
        @rtransform :duration = quantize(:duration, dt)
        wrap_counts(duration = dt:dt:maxt)
    end
end

function optimize_duration_noise(model_df::DataFrame, human_fixations::DataFrame)
    model = duration_hist(make_fixations(model_df); nonfinal=true)
    target = duration_hist(human_fixations; nonfinal=true)
    optimize_ndt(model, target)
end

function add_duration_noise!(df, d)
    for x in df.presentation_times
        x .+= rand(d, length(x))
    end
end

# %% ==================== effect fitting ====================

function compute_effects(name, make_policies, prms; N=100000, kws...)
    compute_cached("$(name)_effects_$N", prms; kws...) do prm
        exp2_effects(make_policies, prm, N)
    end
end

function exp2_effects(make_policies::Function, prm::NamedTuple, N=100_000)
    sim = simulate_exp2(make_policies, prm, N)
    ndt = Gamma(prm.α_ndt, prm.θ_ndt)
    add_duration_noise!(sim, ndt)
    exp2_effects(make_trials(sim), make_fixations(sim))
end

function exp2_effects(trials, fixations)
    fix = @chain fixations begin
        @rsubset :response_type == "correct"
        @rtransform :final = :presentation == :n_pres
    end

    pretest_accuracy = mean(trials.pretest_accuracy_first)
    accuracy = mean(trials.response_type .== "correct")
    rt = mean(@rsubset(trials, :response_type == "correct").rt)
    nfix = counts(@rsubset(trials, :response_type == "correct").n_pres, 1:10)
    duration = mean(fix.duration)

    # nfix[3] ≥ 30 || return missing

    prop_first = @chain trials begin
        @rsubset :response_type == "correct" && :n_pres ≥ 2
        @rtransform :rel_pretest = :pretest_accuracy_first - :pretest_accuracy_second
        @rtransform :prop_first = :total_first / (:total_first + :total_second)
        @regress prop_first ~ rel_pretest
        get_coef
    end

    final = @regress(fix, duration ~ final) |> get_coef

    fixated = @chain fix begin
        @rsubset !:final
        @rtransform :fixated = isodd(:presentation) ? :pretest_accuracy_first : :pretest_accuracy_second
        @regress duration ~ fixated
        get_coef
    end

    nonfixated = @chain fix begin
        @rsubset !:final && :presentation ≥ 2
        @rtransform :nonfixated = iseven(:presentation) ? :pretest_accuracy_first : :pretest_accuracy_second
        @regress duration ~ -nonfixated
        get_coef
    end

    final_dist = @chain fix begin
        @rsubset :final
        @with (gamma=fit(Gamma, :duration), normal=fit(Normal, :duration))
    end

    nonfinal_dist = @chain fix begin
        @rsubset !:final
        @with (gamma=fit(Gamma, :duration), normal=fit(Normal, :duration))
    end

    (;accuracy, pretest_accuracy, rt, nfix, duration, prop_first, final, final_dist, nonfinal_dist, fixated, nonfixated)
end

