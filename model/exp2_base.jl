# %% ==================== simulate ====================

function exp2_mdp(prm; maxt=MAX_TIME)
    time_cost = @isdefined(PRETEST_COST) ? (MS_PER_SAMPLE / MAX_TIME) * .25 : 0
    m2 = MetaMDP{2}(;allow_stop=true, max_step=Int(maxt / MS_PER_SAMPLE), miss_cost=2,
        prm.threshold, prm.noise,
        sample_cost=prm.sample_cost + time_cost, prm.switch_cost,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end

function simulate_exp2(make_policies, prm::NamedTuple, N=1_000_000; kws...)
    simulate_exp2(make_policies(prm)..., N; prm.within_σ, prm.between_σ)
end

function simulate_exp2(pre_pol, crit_pol, N=1_000_000; within_σ, between_σ)
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
            :trial_id = 1:nrow(df)
            :first_pres_time = safeindex(:presentation_times, 1)
            :second_pres_time = safeindex(:presentation_times, 2)
            :third_pres_time = safeindex(:presentation_times, 3)
            :last_pres_time = :presentation_times[end]
            :n_pres = length(:presentation_times)
            :total_first = sum(:presentation_times[1:2:end])
            :total_second = sum(:presentation_times[2:2:end])
            :wid = "optimal"
        end
        @transform :rt = :total_first .+ :total_second
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
