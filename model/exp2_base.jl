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

function make_hist(trials::DataFrame; dt=MS_PER_SAMPLE, maxt=MAX_TIME)
    @chain trials begin
        @rsubset :response_type == "correct"
        # @rsubset :response_type in ("correct", "empty")
        @rtransform! :response = (
            :response_type == "empty" ? "skip" :
            :choose_first ? "first" :
            "second"
        )
        @rtransform! :n_pres = clip(:n_pres, 0, 5)
        @rtransform! :rt = quantize(:rt, dt)
        wrap_counts(
            rt=dt:dt:maxt, 
            response=["first", "second"], 
            # response=["skip", "first", "second"], 
            n_pres=1:5,
            pretest_accuracy_first=0:0.5:1,
            pretest_accuracy_second=0:0.5:1,
        ) 
        normalize!
    end
end

# max_fix_time = @chain human_fixations begin
#     @rsubset :presentation == :n_pres
#     @with quantile(:duration, 0.95)
#     quantize(100)
# end  -> 4600 (hardcoding this)

# function make_hist(fixations::DataFrame; dt=MS_PER_SAMPLE, maxt=4600)
#     @chain fixations begin
#         @rsubset :response_type == "correct"
#         @rtransform! begin 
#             :fixated = isodd(:presentation) ? :pretest_accuracy_first : :pretest_accuracy_second
#             :nonfixated = iseven(:presentation) ? :pretest_accuracy_first : :pretest_accuracy_second
#             :duration = clip(quantize(:duration, dt), 0, maxt)
#             :final = :presentation == :n_pres
#             :presentation = clip(:presentation, 0, 4)
#         end
#         wrap_counts(
#             duration=dt:dt:maxt, 
#             # response=["first", "second"], 
#             final=[false,true],
#             presentation=1:4,
#             fixated=0:0.5:1,
#             nonfixated=0:0.5:1,
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

function compute_histograms(name, make_policies, prms; N=100000, read_only=false, enable_cache=true)
    compute_cached("exp2_$(name)_histograms_$N", prms) do prm
        make_hist(make_trials(simulate_exp2(make_policies, prm, N)))
    end
end

function compute_loss(histograms, prms; sort=true)
    tbl = DataFrame(prms)
    results = @showprogress "loss " pmap(histograms, prms) do model_hist, prm
        ismissing(model_hist) && return [Inf, 1000., 1000.]  # 1000 gives super long RT (a warning flag)
        if hasfield(typeof(prm), :α_ndt)
            (;α_ndt, θ_ndt) = prm
            lk = likelihood(model_hist, human_hist, Gamma(α_ndt, θ_ndt))
            [lk, α_ndt, θ_ndt]
        else
            @assert false
            res = optimize_ndt(model_hist, human_hist)
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
