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

# %% ==================== likelihood stuff ====================

function smooth_uniform!(x, ε::Float64=1e-6)
    x .*= (1 - ε)
    x .+= (ε / length(x))
    x
end

function convolve!(result::AbstractVector{T}, pdf_x::AbstractVector{T}, pdf_y::AbstractVector{T}) where T
    for r in axes(pdf_x, 1)
        result[r] = sum(1:r) do x
            y = r - x
            @inbounds pdf_x[x] * pdf_y[y + 1]
        end
    end
end
# annoying that we have to write each version out, see https://github.com/JuliaLang/julia/issues/29146
function convolve!(result::AbstractArray{T, 2}, pdf_x::AbstractArray{T, 2}, pdf_y::AbstractVector{T}) where T
    for i in axes(pdf_x, 2)
        convolve!(@view(result[:, i]), @view(pdf_x[:, i]), pdf_y)
    end
end
function convolve!(result::AbstractArray{T, 3}, pdf_x::AbstractArray{T, 3}, pdf_y::AbstractVector{T}) where T
    for j in axes(pdf_x, 3), i in axes(pdf_x, 2)
        convolve!(@view(result[:, i, j]), @view(pdf_x[:, i, j]), pdf_y)
    end
end
function convolve!(result::AbstractArray{T, 4}, pdf_x::AbstractArray{T, 4}, pdf_y::AbstractVector{T}) where T
    for k in axes(pdf_x, 4), j in axes(pdf_x, 3), i in axes(pdf_x, 2)
        convolve!(@view(result[:, i, j, k]), @view(pdf_x[:, i, j, k]), pdf_y)
    end
end

function convolve!(result, pdf_x::KeyedArray, y::Distribution)
    pdf_y = diff([0; cdf(y, axiskeys(pdf_x, 1))])
    convolve!(result, pdf_x, pdf_y)
end

function likelihood(model, target, ndt, tmp=zeros(size(model)); ε=1e-5)
    convolve!(tmp, model, ndt)
    smooth_uniform!(tmp, ε)
    crossentropy(target, tmp)
end

function optimize_ndt(model::KeyedArray, target::KeyedArray; ε=1e-5)
    tmp = zeros(size(model))
    optimize([10., 10.]) do x
        any(xi < 0 for xi in x) && return Inf
        likelihood(model, target, Gamma(x...), tmp; ε)
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
