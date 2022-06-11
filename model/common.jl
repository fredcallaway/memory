include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("box.jl")

using Optim
using ProgressMeter

const MAX_TIME = 15000
const MS_PER_SAMPLE = 100
const MAX_STEP = Int(MAX_TIME / MS_PER_SAMPLE)

quantize(x, q=MS_PER_SAMPLE) = q * cld(x, q)

function smooth_uniform!(x, ε::Float64=1e-6)
    x .*= (1 - ε)
    x .+= (ε / length(x))
    x
end

function mean_error(f, x, y)
    size(x) == size(y) || return Inf
    l = mean(f.(x .- y))
    isfinite(l) || return Inf
    l
end

squared(x) = x^2
mae(x, y) = mean_error(abs, x, y)
mse(x, y) = mean_error(squared, x, y)


load_data(name) = CSV.read("../data/processed/$name.csv", DataFrame, missingstring="NA")
stringify(nt::NamedTuple) = replace(string(map(x->round(x; digits=8), nt::NamedTuple)), ([" ", "(", ")"] .=> "")...)

function sample_strengths(pol, N=10000; between_σ, within_σ)
    @assert pol.m.prior.σ^2 ≈ between_σ^2 + within_σ^2
    underlying_strength_dist = Normal(pol.m.prior.μ, between_σ)
    trial_strength_shift = Normal(0, within_σ)

    map(1:N) do i
        underlying_strength = rand(underlying_strength_dist)
        pretest_accuracy = 2 \ mapreduce(+, 1:2) do i
            s = (underlying_strength + rand(trial_strength_shift),)
            simulate(pol; s).b.focused == 1
        end
        strength = underlying_strength + rand(trial_strength_shift)
        (strength, pretest_accuracy)
    end
end

function pretest_mdp(prm)
    time_cost = @isdefined(PRETEST_COST) ? (MS_PER_SAMPLE / MAX_TIME) * .25 : 0
    MetaMDP{1}(;allow_stop=true, max_step=MAX_STEP, miss_cost=1,
        prm.threshold, prm.noise, sample_cost=prm.sample_cost + time_cost,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end

function empirical_distribution(x)
    fit(DiscreteNonParametric, max.(1, round.(Int, x ./ MS_PER_SAMPLE)))
end

function convolve!(result::AbstractArray{Float64, 1}, p::KeyedArray{Float64, 1}, d::Distribution)
    pd = diff([0; cdf(d, axiskeys(p, 1))])
    for z in axes(p, 1)
        result[z] = sum(1:z) do k
            y = z - k
            @inbounds p[k] * pd[y + 1]
        end
    end
end

function convolve!(result::AbstractArray{Float64, 2}, p::KeyedArray{Float64, 2}, d::Distribution)
    pd = diff([0; cdf(d, axiskeys(p, 1))])
    for j in axes(p, 2), z in axes(p, 1)
        result[z, j] = sum(1:z) do k
            y = z - k
            @inbounds p[k, j] * pd[y + 1]
        end
    end
end

function convolve!(result::AbstractArray{Float64, 4}, p::KeyedArray{Float64, 4}, d::Distribution)
    pd = diff([0; cdf(d, axiskeys(p, 1))])
    for h in axes(p, 4), i in axes(p, 3), j in axes(p, 2), z in axes(p, 1)
        result[z, j, i, h] = sum(1:z) do k
            y = z - k
            @inbounds p[k, j, i, h] * pd[y + 1]
        end
    end
end

function optimize_ndt(model::KeyedArray, target::KeyedArray; ε=1e-5)
    X = zeros(size(model))
    optimize([10., 10.]) do x
        any(xi < 0 for xi in x) && return Inf
        convolve!(X, model, Gamma(x...))
        smooth_uniform!(X, ε)
        crossentropy(target, X)
    end
end

function optimize_stopping_model(trials, α_ndt, θ_ndt; ε=1e-5, dt=MS_PER_SAMPLE, maxt=MAX_TIME)
    human = @chain trials begin
        @rsubset :response_type == "empty" && :rt < MAX_TIME
        @rtransform :rt = quantize(:rt, dt)
        wrap_counts(rt = dt:dt:maxt)
        normalize!
    end

    ndt_dist = Gamma(α_ndt, θ_ndt)
    ndt_only = copy(human)
    ndt_only .= diff([0; cdf(ndt_dist, ndt_only.rt)])

    # RT = NDT + stop_time; NDT is fixed, so we can optimize the stopping dist
    # in the same way we optimize NDT for the optimal model.
    α, θ = optimize_ndt(ndt_only, human; ε).minimizer
    Gamma(α, θ / MS_PER_SAMPLE)  # convert to units of samples
end
