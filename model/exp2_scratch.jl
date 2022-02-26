
# %% ==================== Manual ====================

# prm = (drift_μ = 0.6, drift_σ = 0.8, threshold = 5, sample_cost = 0.04, noise = 1.3, switch_cost=.01)
# prm = (drift_μ = 0.5, drift_σ = 0.6, threshold = 4, sample_cost = 0.01, noise = 1.0, switch_cost=.01)
prm = (drift_μ = 0.9, drift_σ = 1.1, threshold = 7, sample_cost = 0.002, 
    noise = 1.4, switch_cost=.002, strength_drift_μ=0, strength_drift_σ=1e-9)

@time df = simulate_optimal(prm)

# counts(make_trials(df).n_pres, 1:4) ./ nrow(df)
# counts(trials.n_pres, 1:4) ./ nrow(trials)
df |> make_trials |> CSV.write("results/exp2/optimal_trials.csv")
df |> make_fixations |> CSV.write("results/exp2/optimal_fixations.csv")

# %% ==================== Fit ====================

@everywhere function twocue_metrics(trials, fixations)
    trials = @rsubset(trials, :response_type == "correct")
    fixations = @rsubset(fixations, :response_type == "correct")

    avg_ptime = @chain fixations begin
        @rsubset(:presentation != :n_pres)
        @by(:wid, :mean=mean(:duration), :sd=std(:duration))
    end

    z_durations = @chain fixations begin
        leftjoin(avg_ptime, on=:wid)
        @rsubset(!isnan(:sd))
        @rtransform(:rel_acc = :pretest_accuracy_first - :pretest_accuracy_second)
        @rsubset(:presentation <= 3 && :presentation != :n_pres)
        @rtransform(:duration_z = (:duration - :mean) / :sd)
        @by([:presentation, :rel_acc], :x = mean(:duration_z))
        wrapdims(:x, :presentation, :rel_acc)
        sortkeys
    end
    (;
        z_durations,
         nfix_hist = counts(trials.n_pres, 1:4) ./ nrow(trials),
         p_correct = mean(trials.response_type .== "correct"),
    )
end
@everywhere twocue_metrics(df) = compute_metrics(make_trials(df), make_fixations(df))

target = compute_metrics(trials, fixations);
@everywhere target = $target

@everywhere squared(x) = x^2
@everywhere function loss(pred)
    nfix_loss = crossentropy(target.nfix_hist, pred.nfix_hist)
    isfinite(nfix_loss) || return Inf
    duration_loss = mean(squared.(pred.z_durations .- target.z_durations))
    nfix_loss + duration_loss
end