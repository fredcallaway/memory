
# const ss_human = exp2_sumstats(human_trials, human_fixations);

function compute_sumstats(name, make_policies, prms; N=100000, read_only=false, enable_cache=true)
    dir = "cache/$(RUN)_exp2_$(name)_sumstats_$N"
    mkpath(dir)
    map = read_only ? asyncmap : pmap
    @showprogress map(prms) do prm
        cache("$dir/$(hash(prm))"; read_only, disable=!enable_cache) do
            sim = simulate_exp2(make_policies, prm, N)
            ndt = Gamma(prm.α_ndt, prm.θ_ndt)
            add_duration_noise!(sim, ndt)
            exp2_sumstats(make_trials(sim), make_fixations(sim))
        end
    end;
end


function exp2_sumstats(trials, fixations)
    accuracy = mean(trials.response_type .== "correct")

    tri = @chain trials begin
        @rsubset :response_type == "correct" 
        groupby([:wid, :pretest_accuracy_first, :pretest_accuracy_second, :choose_first])
        @combine begin
            :rt_μ = mean(:rt)
            :rt_σ = std(:rt)
            :prop_first = mean(:total_first ./ (:total_first .+ :total_second))
            :total_first = mean(:total_first)
            :total_second = mean(:total_second)
            :n = length(:rt)
        end
    end

    fix = @chain fixations begin
        @rsubset :response_type == "correct"
        @rtransform :final = :presentation == :n_pres
        @rtransform :fixfirst = isodd(:presentation)
        groupby([:wid, :pretest_accuracy_first, :pretest_accuracy_second, :final, :choose_first])
        @combine begin
            :duration_μ = mean(:duration) 
            :duration_σ = std(:duration) 
            :n = length(:duration)
        end
    end

    (;accuracy, tri, fix)
end