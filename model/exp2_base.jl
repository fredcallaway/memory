function exp2_mdp(prm)
    time_cost = (ms_per_sample / 1000) * (.25 / 15)
    m2 = MetaMDP{2}(;allow_stop=true, max_step=60, miss_cost=2,
        prm.threshold, prm.noise,
        sample_cost=prm.sample_cost + time_cost, prm.switch_cost,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end

function simulate_exp2(make_policies, prm::NamedTuple, N=500000; kws...)
    strength_drift = Normal(prm.strength_drift_μ, prm.strength_drift_σ)
    simulate_exp2(make_policies(prm)..., N; strength_drift)
end

function simulate_exp2(pre_pol, crit_pol, N=500000; 
                       strength_drift=Normal(0, 1e-9), duration_noise=Gamma(1e-9,1e-9))
    strengths = sample_strengths(pre_pol,  2N; strength_drift)
    pairs = map(1:2:2N) do i
        s1, pretest_accuracy_first = strengths[i]
        s2, pretest_accuracy_second = strengths[i+1]
        (;pretest_accuracy_first, pretest_accuracy_second), (s1, s2)
    end

    map(pairs) do (pretest_accuracy, s)
        sim = simulate(crit_pol; s, fix_log=FullFixLog())
        presentation_times = sim.fix_log.fixations .* float(ms_per_sample)
        # presentation_times .+= (crit_pol.m.switch_cost / crit_pol.m.sample_cost) * ms_per_sample
        presentation_times .+= rand(duration_noise, length(presentation_times))
        (;response_type = sim.b.focused == -1 ? "empty" : "correct",
          choose_first = sim.b.focused == 1,
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
        select(names(trials))
    end
end

function make_fixations(df)
    @chain df begin
        @transform(:trial_id = 1:nrow(df))
        @rtransform(:n_pres = length(:presentation_times))
        @rtransform(:presentation = 1:(:n_pres), :wid = "optimal")
        DataFrames.flatten([:presentation_times, :presentation])
        DataFrames.rename(:presentation_times => :duration)
        select(names(fixations))
    end
end


# %% ==================== summary statistics ====================

function unroll_trial!(P, durations, choice; dt)
    max_step = size(P, 1)
    breaks = round.(Int, cumsum(durations) / dt)

    # denom[1:min(breaks[end], max_step)] .+= 1
    start = 1
    for (i, stop) in enumerate(breaks)
        stop = min(stop, max_step)
        fix = iseven(i) ? 2 : 1
        P[start:stop, fix] .+= 1
        start = stop + 1
        start > max_step && return
    end
    P[start:max_step, choice+2] .+= 1
end

function unroll_time(fixations; dt=ms_per_sample, maxt=15000)
    @chain fixations begin
        # @rsubset :response_type == "correct"
        @rtransform :rel_pretest_accuracy = :pretest_accuracy_first - :pretest_accuracy_second
        groupby(:rel_pretest_accuracy)
        combine(_) do d
            P = zeros(Int(maxt/dt), 4)
            grp = groupby(d, :trial_id)
            for d in grp
                choice = 2 - d.choose_first[1]
                unroll_trial!(P, d.duration, choice; dt)
            end
            P ./= length(grp)
            Ref(P)  # prevents unrolling the array
        end
        @orderby :rel_pretest_accuracy
        @with combinedims(:x1)
        @catch_missing KeyedArray(_,
            time=dt:dt:maxt, 
            event=[:fix1, :fix2, :choose1, :choose2], 
            rel_pretest_accuracy=-1:.5:1
        )
    end 
end

function exp2_sumstats(trials, fixations)
    accuracy = mean(trials.response_type .== "correct")

    tri = @chain trials begin
        # @rsubset :response_type == "correct" 
        groupby([:wid, :pretest_accuracy_first, :pretest_accuracy_second])
        @combine begin
            :rt_μ = mean(:rt)
            :rt_σ = std(:rt)
            :choose_first = mean(:choose_first)
            :prop_first = mean(:total_first ./ (:total_first .+ :total_second))
            :total_first = mean(:total_first)
            :total_second = mean(:total_second)
            :n = length(:rt)
        end
    end

    fix = @chain fixations begin
        # @rsubset :response_type == "correct"
        @rtransform :final = :presentation == :n_pres
        @rtransform :presentation = min(:presentation, 10)
        groupby([:wid, :presentation, :pretest_accuracy_first, :pretest_accuracy_second, :final, :choose_first])
        @combine begin
            :duration_μ = mean(:duration) 
            :duration_σ = std(:duration) 
            :n = length(:duration)
        end
    end

    unrolled = unroll_time(fixations)
    (;accuracy, tri, fix, unrolled)
end
