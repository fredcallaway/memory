
# %% ==================== summary statistics ====================

@everywhere function unroll_trial!(P, durations, choice; dt)
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

@everywhere function unroll_time(fixations; dt=ms_per_sample, maxt=15000)
    @chain fixations begin
        @rsubset :response_type == "correct"
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

@everywhere function exp2_sumstats(trials, fixations)
    accuracy = mean(trials.response_type .== "correct")

    tri = @chain trials begin
        @rsubset :response_type == "correct" 
        groupby([:wid, :pretest_accuracy_first, :pretest_accuracy_second])
        @combine begin
            :rt_μ = mean(:rt)
            :rt_σ = std(:rt)
            :choose_first = mean(:choose_first)
            :prop_first = mean(:total_first ./ :total_second)
            :total_first = mean(:total_first)
            :total_second = mean(:total_second)
            :n = length(:rt)
        end
    end

    fix = @chain fixations begin
        @rsubset :response_type == "correct"
        @rtransform :final = :presentation == :n_pres
        @rtransform :presentation = min(:presentation, 10)
        groupby([:wid, :presentation, :pretest_accuracy_first, :pretest_accuracy_second, :final])
        @combine begin
            :duration_μ = mean(:duration) 
            :duration_σ = std(:duration) 
            :n = length(:duration)
        end
    end

    unrolled = unroll_time(fixations)
    (;accuracy, tri, fix, unrolled)
end

# struct Exp2Loss
#     trials::DataFrame
#     fixations::DataFrame
#     target::NamedTuple
# end

target = exp2_sumstats(trials, fixations);

# %% ==================== loss ====================

function nonfinal_duration(metric)    
    X = @chain metric.fix begin
        @rsubset :presentation > 1 && !:final
        groupby(:wid)
        @transform @astable begin
            μ, σ = pooled_mean_std(:n, :duration_μ, :duration_σ)
            :z = (:duration_μ .- μ) / σ
        end
        @rsubset !isnan(:z)
        @rtransform @astable begin
            x = (:pretest_accuracy_first, :pretest_accuracy_second)
            f, n = iseven(:presentation) ? reverse(x) : x
            :fixated = f
            :nonfixated = n
            :relative = f - n
        end
        @bywrap [:relative] mean(:z, Weights(:n))
    end
end

_human_nonfinal = nonfinal_duration(target)
function fix_loss(pred)
    p = nonfinal_duration(pred)
    size(_human_nonfinal) == size(p) || return Inf
    sum(squared.(_human_nonfinal .- p))
end