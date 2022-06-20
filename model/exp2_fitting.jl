
# const ss_human = exp2_sumstats(human_trials, human_fixations);

function compute_sumstats(name, make_policies, prms; read_only = false)
    dir = "cache/$(RUN)_exp2_$(name)_sumstats"
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

function unroll_trial!(P, durations, choice; dt)
    max_step = size(P, 1)
    breaks = round.(Int, cumsum(durations) / dt)

    start = 1
    for (i, stop) in enumerate(breaks)
        stop = min(stop, max_step)
        fix = iseven(i) ? 2 : 1
        P[start:stop, fix] .+= 1
        start = stop + 1
        start > max_step && return
    end
    if ismissing(choice)
        P[start:max_step, 5] .+= 1        
    else
        P[start:max_step, choice+2] .+= 1
    end
end

function unroll_time(fixations; dt=MS_PER_SAMPLE, maxt=MAX_TIME)
    @chain fixations begin
        @rsubset :response_type == "correct"
        groupby([:pretest_accuracy_first, :pretest_accuracy_second, :choose_first])
        combine(_) do d
            P = zeros(Int(maxt/dt), 5)
            grp = groupby(d, :trial_id)
            for d in grp
                choice = 2 - d.choose_first[1]
                unroll_trial!(P, d.duration, choice; dt)
            end
            P ./= length(grp)
            Ref(P)  # prevents unrolling the array
        end
        DataFrames.rename(:x1 => :timecourse)
        @orderby :pretest_accuracy_first :pretest_accuracy_second
        # @rtransform :rel_pretest_accuracy = :pretest_accuracy_first - :pretest_accuracy_second
        # @with combinedims(:x1)
        # @catch_missing KeyedArray(_,
        #     time=dt:dt:maxt, 
        #     event=[:fix1, :fix2, :choose1, :choose2, :skip],
        #     rel_pretest_accuracy=-1:.5:1
        # )
    end 
end

function exp2_sumstats(trials, fixations)
    accuracy = mean(trials.response_type .== "correct")

    # tri = @chain trials begin
    #     @rsubset :response_type == "correct" 
    #     groupby([:wid, :pretest_accuracy_first, :pretest_accuracy_second, :choose_first])
    #     @combine begin
    #         :rt_μ = mean(:rt)
    #         :rt_σ = std(:rt)
    #         :prop_first = mean(:total_first ./ (:total_first .+ :total_second))
    #         :total_first = mean(:total_first)
    #         :total_second = mean(:total_second)
    #         :n = length(:rt)
    #     end
    # end

    tri = @chain trials begin
        groupby([:wid, :pretest_accuracy_first, :pretest_accuracy_second, :choose_first, :n_pres])
        @combine :n = length(:rt)
    end

    fix = @chain fixations begin
        @rsubset :response_type == "correct"
        @rtransform :final = :presentation == :n_pres
        @rtransform :presentation = min(:presentation, 10)
        groupby([:wid, :presentation, :pretest_accuracy_first, :pretest_accuracy_second, :final, :choose_first])
        @combine begin
            :duration_μ = mean(:duration) 
            :duration_σ = std(:duration) 
            :n = length(:duration)
        end
    end

    nonfinal = @chain fixations begin
        @rsubset :response_type == "correct"
        nonfinal_duration_hist
    end

    unrolled = unroll_time(fixations)

    (;accuracy, tri, fix, unrolled, nonfinal)
end