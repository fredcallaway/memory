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
function initialize_keyed(val; keys...)
    KeyedArray(fill(val, (length(v) for (k, v) in keys)...); keys...)
end

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

macro bywrap(x, what, val, default=missing)
    arg = :(:_val = $val)
    esc(quote
        b = $(DataFramesMeta.by_helper(x, what, arg))
        what_ = $what isa Symbol ? ($what,) : $what
        wrapdims(b, :_val, what_..., sort=true; default=$default)
    end)
end

function sample_strengths(pol, N=10000; strength_noise=0.)

    prenoise_prior = mutate(pol.m.prior, σ=√(pol.m.prior.σ^2 - strength_noise^2))
    noise = Normal(0, strength_noise)

    map(1:N) do i
        prenoise_strength = rand(prenoise_prior)
        pretest_accuracy = 2 \ mapreduce(+, 1:2) do i
            s = (prenoise_strength + rand(noise),)
            simulate(pol; s).b.focused == 1
        end
        strength = prenoise_strength + rand(noise)
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

function compute_loss(loss, sumstats, prms)
    ismissing(sumstats) && return Inf
    tbl = DataFrame(prms)
    tbl.loss = @showprogress "loss" pmap(loss, sumstats);
    tbl.ss = sumstats
    sort!(tbl, :loss)
end


