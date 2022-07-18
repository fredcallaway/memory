
# const ss_human = exp1_effects(human_trials, human_fixations);

function compute_effects(name, make_policies, prms; N=100000, read_only=false, enable_cache=true)
    compute_cached("exp1_$(name)_effects_$N", prms) do prm
        exp1_effects(make_policies, prm, N)
    end
end

function exp1_effects(make_policies::Function, prm::NamedTuple, N=100_000)
    sim = simulate_exp1(make_policies, prm, N)
    ndt = Gamma(prm.α_ndt, prm.θ_ndt)
    sim.rt = sim.rt .+ rand(ndt, nrow(sim))
    exp1_effects(sim)
end

function exp1_effects(trials)
    accuracy = mean(trials.response_type .== "correct")

    empty_judgement = @chain trials begin
        @rsubset :response_type == "empty"
        @regress rt ~ judgement
        get_coef
    end
    
    empty_pretest = @chain trials begin
        @rsubset :response_type == "empty"
        @regress rt ~ pretest_accuracy
        get_coef
    end
    (;accuracy, empty_judgement, empty_pretest)
end
