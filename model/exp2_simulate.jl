
function simulate_pretest(prm, N=10000)
    m = MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=1,
        prm.threshold, prm.sample_cost, prm.noise,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
    pol = OptimalPolicy(m; dv=m.threshold*.02)
    
    mapreduce(vcat, 1:N) do i
        s = sample_state(pol.m)
        map(1:2) do j
            sim = simulate(pol; s, fix_log=RTLog())
            (;
                wid="optimal",
                word=i,
                strength=only(s),
                response_type = sim.b.focused == -1 ? "empty" : "correct",
                rt=sim.fix_log.rt * ms_per_sample,
            )
        end
    end |> DataFrame
end

function simulate_optimal(prm::NamedTuple, N=100000; kws...)
    m1 = MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=1,
        prm.threshold, prm.sample_cost, prm.switch_cost, prm.noise,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
    pol1 = OptimalPolicy(m1)

    m2 = MetaMDP{2}(;allow_stop=true, max_step=60, miss_cost=2,
        prm.threshold, prm.sample_cost, prm.switch_cost, prm.noise,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
    pol2 = OptimalPolicy(m2)
    strength_drift = Normal(prm.strength_drift_μ, prm.strength_drift_σ)
    make_frame(pol1, pol2, N; strength_drift)
end


function make_frame(pol1, pol2, N=10000; strength_drift=Normal(0, 1e-9), duration_noise=Gamma(1e-9,1e-9))
    strengths = sample_strengths(pol1,  2N; strength_drift)
    pairs = map(1:2:2N) do i
        s1, pretest_accuracy_first = strengths[i]
        s2, pretest_accuracy_second = strengths[i+1]
        (;pretest_accuracy_first, pretest_accuracy_second), (s1, s2)
    end

    sims = map(pairs) do (pretest_accuracy, s)
        sim = simulate(pol2; s, fix_log=FullFixLog())
        presentation_times = sim.fix_log.fixations .* float(ms_per_sample)
        # presentation_times .+= (pol2.m.switch_cost / pol2.m.sample_cost) * ms_per_sample
        presentation_times .+= rand(duration_noise, length(presentation_times))
        (;response_type = sim.b.focused == -1 ? "empty" : "correct",
          choose_first = sim.b.focused == 1,
          pretest_accuracy..., 
          strength_first = s[1], strength_second = s[2],
          presentation_times
         )
    end
    DataFrame(sims[:])
end

function make_trials(df)
    safeindex(x, i) = length(x) < i ? NaN : x[i]
    @chain df begin
         @rtransform(
            :first_pres_time = safeindex(:presentation_times, 1),
            :second_pres_time = safeindex(:presentation_times, 2),
            :third_pres_time = safeindex(:presentation_times, 3),
            :last_pres_time = :presentation_times[end],
            :n_pres = length(:presentation_times),
            :total_first = sum(:presentation_times[1:2:end]),
            :total_second = sum(:presentation_times[2:2:end]),
            :wid = "optimal"
        )
        @transform :rt = :total_first .+ :total_second
        select(setdiff(names(trials)))
    end
end

function make_fixations(df)
    @chain df begin
        @transform(:trial_id = 1:nrow(df))
        @rtransform(:n_pres = length(:presentation_times))
        @rtransform(:presentation = 1:(:n_pres), :wid = "optimal")
        DataFrames.flatten([:presentation_times, :presentation])
        DataFrames.rename(:presentation_times => :duration)
        select(setdiff(names(fixations)))
    end
end
