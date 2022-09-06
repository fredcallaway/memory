
# const ss_human = exp2_effects(human_trials, human_fixations);

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