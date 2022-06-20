
# const ss_human = exp2_sumstats(human_trials, human_fixations);

function compute_sumstats(name, make_policies, prms; N=100000, read_only=false, enable_cache=true)
    dir = "cache/$(RUN)_exp2_$(name)_sumstats_$N"
    mkpath(dir)
    map = read_only ? asyncmap : pmap
    @showprogress map(prms) do prm
        cache("$dir/$(hash(prm))"; read_only, disable=!enable_cache) do
            exp2_sumstats(make_policies, prm, N)
        end
    end;
end

function exp2_sumstats(make_policies::Function, prm::NamedTuple, N=100_000)
    sim = simulate_exp2(make_policies, prm, N)
    ndt = Gamma(prm.α_ndt, prm.θ_ndt)
    add_duration_noise!(sim, ndt)
    exp2_sumstats(make_trials(sim), make_fixations(sim))
end

robust_median(x) = length(x) < 30 ? NaN : median(x)

function exp2_sumstats(trials, fixations)
    accuracy = mean(trials.response_type .== "correct")
    nfix = counts(@rsubset(trials, :response_type == "correct").n_pres, 1:10)

    prop_first = @chain trials begin
        @rsubset :response_type == "correct" && :n_pres ≥ 2
        @rtransform :rel_pretest = :pretest_accuracy_first - :pretest_accuracy_second
        @rtransform :prop_first = :total_first / (:total_first + :total_second)
        @bywrap :rel_pretest robust_median(:prop_first)
    end

    fix = @chain fixations begin
        @rsubset :response_type == "correct"
        @rtransform :final = :presentation == :n_pres
    end

    final = @bywrap fix :final robust_median(:duration)

    fixated = @chain fix begin
        @rsubset !:final
        @rtransform :fixated = isodd(:presentation) ? :pretest_accuracy_first : :pretest_accuracy_second
        @bywrap :fixated robust_median(:duration)
    end

    nonfixated = @chain fix begin
        @rsubset !:final && :presentation ≥ 2
        @rtransform :nonfixated = iseven(:presentation) ? :pretest_accuracy_first : :pretest_accuracy_second
        @bywrap :nonfixated robust_median(:duration)
    end

    (;accuracy, nfix, prop_first, final, fixated, nonfixated)
end