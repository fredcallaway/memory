using Revise
using DataFrames
using CSV
using Query
using Turing
using Logging


includet("binomial_accumulator.jl")
include("utils.jl")
include("figure.jl")

# %% --------
version = "v4.0"

simple = CSV.read("../data/$version/simple-recall.csv", DataFrame) |> @mutate(
    target = split(_.image, "/")[4]
) |> @select(-:image) |> DataFrame

afc = CSV.read("../data/$version/afc.csv", DataFrame) |> @mutate(
    target = split(_.target, "/")[1],
    lure = split(_.lure, "/")[1]
) |> DataFrame

# %% --------

afc1 = afc |> @filter(_.block > 1) |> @groupby(_.wid) |> @drop(3) |> first |> DataFrame

# %% --------

simple1 = simple |> @groupby(_.wid) |> @drop(3) |> first |> @filter(_.response_type == "correct") |> DataFrame

x = float.(simple1.typing_rt)
fit(Gamma, x)

# %% --------


# %% --------
function integer_labels(xs)
    ux = unique(xs)
    labs = Dict(zip(ux, eachindex(xs)))
    [labs[x] for x in xs]
end

afc.wid_i = integer_labels(afc.wid)
df = mapreduce(vcat, groupby(afc, :wid)) do d
    invert((;d.wid_i, word=integer_labels(d.word), d.rt))
end  |> DataFrame
df.rt ./= 25


function rp_to_αθ(r, p)
    μ = ((1 - p) * r) / p
    σ2 = ((1-p) * r) / p^2
    α = μ^2 / σ2
    θ = σ2 / μ
    α, θ
end

quantile(df.rt, [0, .1, .5, .9, 1])

max_rt = mean(df.rt) + 4 * std(df.rt)
df.rt = min.(df.rt, max_rt)

# %% ==================== By participant ====================

function prep_data(df)
    map(groupby(df, :wid_i)) do g
        collect(g.rt)
    end
end

@model function rt_model(data)
    r ~ filldist(Uniform(1, 30), length(data))
    p ~ filldist(Uniform(0.001, .99), length(data))
    for w in eachindex(data)
        α, θ = rp_to_αθ(r[w], p[w])
        for i in eachindex(data[w])
            data[w][i] ~ Gamma(α, θ)
        end
    end
end

data = df |> @filter(_.wid_i < 4) |> DataFrame |> prep_data

# Start sampling.
chain = with_logger(SimpleLogger(stderr, Logging.Error)) do
    sample(rt_model(data), NUTS(), MCMCThreads(), 1000, 8)
end

# %% -------- ALTERNATIVE FORM

@model function rt_model(subj, rt)
    n_subj = maximum(subj)

    r ~ filldist(Uniform(10, 100), n_subj)
    p ~ filldist(Uniform(0.1, 0.9), n_subj)

    for i in 1:size(data, 1)
        s = subj[i]
        α, θ = rp_to_αθ(r[s], p[s])
        rt[i] ~ Gamma(α, θ)
    end
end

data = df |> @filter(_.wid_i < 4) |> DataFrame

sample(rt_model(data.wid_i, data.rt), NUTS(), 100)


# %% ==================== By word ====================
using Parameters

@model function rt_model(subj, word, rt)
    n_subj = maximum(data.wid_i)
    n_word = maximum(data.word)

    r ~ filldist(Uniform(1, 30), n_subj)
    p ~ filldist(Uniform(0.001, 0.99), n_subj, n_word)

    for i in 1:size(data, 1)
        subj = wid_i[i]; w = word[i]
        α, θ = rp_to_αθ(r[subj], p[subj, w])
        rt[i] ~ Gamma(α, θ)
    end
end

# sample(rt_model(data.wid_i, data.word, data.rt), NUTS(), 100)
data = df
chain = with_logger(SimpleLogger(stderr, Logging.Error)) do
    sample(rt_model(data.wid_i, data.word, data.rt), NUTS(), MCMCThreads(), 500, 8)
end

# %% --------
samples = DataFrame(chain)

samples[!, "r[3]"]
# %% --------

# Settings of the Hamiltonian Monte Carlo (HMC) sampler.

ϵ = 0.05
τ = 10

data = df |> @filter(_.wid_i < 4) |> DataFrame |> prep_data

# Start sampling.
logger = SimpleLogger(stderr, Logging.Error) # or e.g. Logging.Warn
chain = with_logger(logger) do
    sample(rt_model(data), NUTS(), MCMCThreads(), 1000, 8)
end

chain

# %% ==================== MLE ====================
using Optim
using StatsFuns: logistic
rt = data[1]

include("box.jl")
space = Box(
    r = (10, 100),
    p = (.1, .9),
)

function nll(rt, r, p)
    α, θ = rp_to_αθ(r, p)
    -sum(logpdf(Gamma(α, θ), rt))
end

res = optimize(zeros(2)) do x
    r, p = space(logistic.(x))
    nll(rt, r, p)
end
space(logistic.(res.minimizer))

# %% ==================== heatmaps ====================



data = df |> @filter(_.wid_i <= 9) |> DataFrame |> prep_data

Xs = map(data) do rt
    X = map(grid(10, space)) do g
       nll(rt, g.r, g.p)
    end
end

figure() do
    ps = map(Xs) do X
        X = copy(X)
        X[X .> 2 * minimum(X)] .= NaN
        heatmap(X)
    end
    plot(ps..., size=(900, 900))
end

# %% --------

rts = data[1]
ress = map(10:10:80) do r
    optimize(0, 1) do p
        nll(rt, r, p)
    end
end

# %% --------
space = Box(
    r = (1, 20),
    p = (.01, .99),
)

rt = data[1]

ress = map(data) do rt
    res = optimize(zeros(2)) do x
        r, p = space(logistic.(x))
        nll(rt, r, p)
    end
    space(logistic.(res.minimizer)), res.minimum
end

ress2 = map(data) do rt
    r = 3
    res = optimize(0, 1) do p
        nll(rt, r, p)
    end
    (;r, p=res.minimizer), res.minimum
end

loss = [x[2] for x in ress]
loss2 = [x[2] for x in ress2]

loss .- loss2

# %% --------
rt = data[3]

p2 = optimize(0, 1) do p
    nll(rt, 2, p)
end |> Optim.minimizer

p5 = optimize(0, 1) do p
    nll(rt, 5, p)
end |> Optim.minimizer

p20 = optimize(0, 1) do p
    nll(rt, 20, p)
end |> Optim.minimizer

# %% --------
figure() do
    plot!(Gamma(rp_to_αθ(2, p2)...))
    plot!(Gamma(rp_to_αθ(5, p5)...))
    plot!(Gamma(rp_to_αθ(20, p5)...))
end




# %% --------
samples = DataFrame(chain)

samples[!, "r[5]"]

# %% --------
figure() do
    plot(chain)
end

# %% --------


@model function rt_model(rt)
    # Our prior belief about the probability of heads in a coin.
    # rt = df.rt
    r ~ Uniform(10, 100)
    p ~ Uniform(0.1, 0.9)
    α, θ = rp_to_αθ(r, p)
    for i in eachindex(rt)
        rt[i] ~ Gamma(α, θ)
    end
end

# %% --------
figure() do
    plot(chain)
end

# %% --------
figure() do
    for (α, θ) in Iterators.product([1, 3], [1, 3])
        plot!(Gamma(α, θ), label="α=$α, θ=$θ")
    end
end



# %% --------

# Import libraries.
using Turing, StatsPlots, Random

# Set the true probability of heads in a coin.
p_true = 0.5

# Iterate from having seen 0 observations to 100 observations.
Ns = 0:100

# Draw data from a Bernoulli distribution, i.e. draw heads or tails.
Random.seed!(12)
data = rand(Bernoulli(p_true), last(Ns))

# Declare our Turing model.
@model function coinflip(y)
    # Our prior belief about the probability of heads in a coin.
    p ~ Beta(1, 1)

    # The number of observations.
    N = length(y)
    for n in 1:N
        # Heads or tails of a coin are drawn from a Bernoulli distribution.
        y[n] ~ Bernoulli(p)
    end
end

# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
iterations = 1000
ϵ = 0.05
τ = 10

# Start sampling.
chain = sample(coinflip(data), HMC(ϵ, τ), iterations)

# Plot a summary of the sampling process for the parameter p, i.e. the probability of heads in a coin.
histogram(chain[:p])
