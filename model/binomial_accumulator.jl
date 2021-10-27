using Parameters
using Distributions
using SplitApplyCombine
using StatsBase
using StatsFuns

@with_kw struct MetaMDP{N}
    threshold::Int = 100
    sample_cost::Float64 = 1.
    switch_cost::Float64 = 0.
    miss_cost::Float64 = 0.
    step_size::Int = 10
    max_step::Int = 100
    allow_stop::Bool = false
    prior::Tuple{Int, Int} = (1, 1)
end
MetaMDP(;kws...) = MetaMDP{2}(;kws...)
getfields(x) = (getfield(x, f) for f in fieldnames(typeof(x)))
id(m::MetaMDP) = join(getfields(m), "-")

State{N} = NTuple{N, Float64}
function sample_state(m::MetaMDP{N})::State{N} where N
    Tuple(rand(Beta(m.prior...), N))
end

struct Belief{N}
    n_step::Int
    focused::Int
    counts::NTuple{N, Tuple{Int, Int}}
end

actions(m::MetaMDP, b::Belief) = m.allow_stop ? (0:n_item(b)) : (1:n_item(b))

function initial_belief(m::MetaMDP{N})::Belief{N} where N
    Belief(0, 1, Tuple(m.prior for i in 1:N))  # note assume focus first item first
end
# terminal_belief(m::MetaMDP, c::Int=-1) = Belief(-1, c, Tuple((-1, -1) for i in 1:n_item(m)))
is_terminal(b) = b.n_step == -1
# max_cost(m::MetaMDP) = m.miss_cost

n_item(m::MetaMDP{N}) where N = N
n_item(b::Belief{N}) where N = N

# Two versions of the dynamics. First one marginalizes over state. Second
# conditions on the true state (which is unknown to the model). Only difference
# is the distribution the number of heads is drawn from, so we separate out
# the constant parts to _results
function results(m::MetaMDP{N}, b::Belief{N}, c::Int) where N
    α, β = b.counts[c]
    head_dist = BetaBinomial(m.step_size, α, β)
    _results(head_dist, m, b, c)
end

function results(m::MetaMDP{N}, b::Belief{N}, s::State{N}, c::Int) where N
    head_dist = Binomial(m.step_size, s[c])
    _results(head_dist, m, b, c)
end

results(m::MetaMDP{N}, b::Belief{N}, s::Nothing, c::Int) where N = results(m, b, c)

function _results(head_dist::Distribution, m::MetaMDP{N}, b::Belief{N}, c::Int) where N
    @assert !is_terminal(b)
    @assert b.n_step < m.max_step
    if c == 0  # give up
        @assert m.allow_stop
        return [(1., Belief(-1, -1, b.counts), -m.miss_cost)]
    end

    cost = (b.n_step == 0 || c != b.focused) ? m.sample_cost + m.switch_cost : m.sample_cost
    map(0:m.step_size) do heads
        α, β = b.counts[c]
        p = pdf(head_dist, heads)
        update = (α + heads, β + m.step_size - heads)
        new_counts = Base.setindex(b.counts, update, c)

        if α + heads >= (m.threshold + m.prior[1])  # hit threshold
            return (p, Belief(-1, c, new_counts), -cost)
        elseif b.n_step == m.max_step - 1  # time out
            return (p, Belief(-1, -1, new_counts), -(cost + m.miss_cost))
        else  # regular old update
            update = (α + heads, β + m.step_size - heads)
            new_counts = Base.setindex(b.counts, update, c)
            return (p, Belief(b.n_step + 1, c, new_counts), -cost)
        end
    end
end
            
# %% ==================== Solution ====================

struct ValueFunction{F}
    m::MetaMDP
    hasher::F
    cache::Dict{UInt64, Float64}
end

function default_hash(b::Belief{1})
    hash(b)
end

function default_hash(b::Belief{2})
    return hash(b)
    not_focused = b.focused == 1 ? 2 : 1
    hash(b.n_step, hash(b.counts[b.focused], hash(b.counts[not_focused])))
end

ValueFunction(m::MetaMDP, hasher::Function) = ValueFunction(m, hasher, Dict{UInt64, Float64}())
ValueFunction(m::MetaMDP) = ValueFunction(m, default_hash)


function Q(V::ValueFunction, b::Belief, c)::Float64
    sum(p * (r + V(b1)) for (p, b1, r) in results(V.m, b, c))
end

Q(V::ValueFunction, b::Belief) = [Q(V,b,c) for c in actions(V.m, b)]


function (V::ValueFunction)(b::Belief)::Float64
    is_terminal(b) && return 0.

    # short circuit if we're doomed to fail
    # steps_left = m.max_step - b.n_step
    # min_needed = m.threshold - steps_left * m.step_size
    # if !any(α >= min_needed for (α, β) in b.counts)
    #     return steps_left * m.sample_cost + m.miss_cost
    # end

    key = V.hasher(b)
    haskey(V.cache, key) && return V.cache[key]
    return V.cache[key] = step_V(V, b)
end

function step_V(V::ValueFunction, b::Belief)
    maximum(Q(V, b, c) for c in actions(V.m, b))
end

# function step_V(V::ValueFunction, b)::Float64
#     # best = term_reward(V.m, b)
#     @fastmath @inbounds for c in 1:n_item(b)
#         val = 0.
#         R = V.m.rewards[c]
#         for i in eachindex(R.p)
#             v = R.support[i]; p = R.p[i]
#             b1 = copy(b)
#             b1[c] = v
#             val += p * (V(b1) - V.m.cost)
#         end
#         if val > best
#             best = val
#         end
#     end
#     best
# end

function Base.show(io::IO, v::ValueFunction)
    print(io, "V")
end


# function solve(m::MetaMDP, h=choose_hash(m))
#     V = ValueFunction(m, h)
#     V(initial_belief(m))
#     V
# end

# function load_V_nomem(i::String)
#     println("Loading V $i")
#     V = deserialize("mdps/V/$i")
#     ValueFunction(V.m, choose_hash(V.m), V.cache)
# end

# @memoize load_V(i::String) = load_V_nomem(i)
# # load_V(m::MetaMDP) = load_V(id(m))

# _lru() = LRU{Tuple{String}, ValueFunction}(maxsize=1)
# @memoize _lru load_V_lru2(i::String) = load_V_nomem(i)

# %% ==================== Policy ====================

function argmaxes(f, x)
    fx = f.(x)
    mfx = maximum(fx)
    x[findall(isequal(mfx), fx)]
end

abstract type Policy end

struct RandomPolicy <: Policy
    m::MetaMDP
end

act(pol::RandomPolicy, b::Belief) = rand(1:n_item(b))


struct RandomSwitchPolicy <: Policy
    m::MetaMDP
    p_switch::Float64
end

function act(pol::RandomSwitchPolicy, b::Belief)
    if b.focused == 0 || rand() < pol.p_switch
        rand(setdiff(1:n_item(b), b.focused))
    else
        b.focused
    end
end

mutable struct SwitchDistributionPolicy{D<:Distribution} <: Policy 
    m::MetaMDP
    dist::D
    time_to_switch::Int
end

SwitchDistributionPolicy(m::MetaMDP, d::Distribution) = SwitchDistributionPolicy(m, d, 0)

function act(pol::SwitchDistributionPolicy, b::Belief)
    if b.n_step == 0 || pol.time_to_switch == 0
        pol.time_to_switch = ceil(Int, rand(pol.dist)) - 1
        rand(setdiff(1:n_item(b), b.focused))
    else
        pol.time_to_switch -= 1
        b.focused
    end
end


struct OptimalPolicy <: Policy
    m::MetaMDP
    V::ValueFunction
end

OptimalPolicy(m::MetaMDP) = OptimalPolicy(m, ValueFunction(m))
OptimalPolicy(V::ValueFunction) = OptimalPolicy(V.m, V)

function act(pol::OptimalPolicy, b::Belief)
    rand(argmaxes(c -> Q(pol.V, b, c), actions(pol.m, b)))
end


struct SoftOptimalPolicy <: Policy
    m::MetaMDP
    V::ValueFunction
    β::Float64
end

SoftOptimalPolicy(m::MetaMDP; β::Real) = SoftOptimalPolicy(m, ValueFunction(m), float(β))
SoftOptimalPolicy(V::ValueFunction; β::Real) = SoftOptimalPolicy(V.m, V, float(β))

function act(pol::SoftOptimalPolicy, b::Belief)
    p = softmax!(map(c-> pol.β * Q(pol.V, b, c), actions(pol.m, b)))
    sample(actions(pol.m, b), Weights(p))
end


function simulate(policy; b=initial_belief(policy.m), s=nothing, save_beliefs=false)
    m = policy.m
    total_cost = 0.
    bs = Belief[]
    cs = Int[]
    while !is_terminal(b)
        c = act(policy, b)
        save_beliefs && push!(bs, b)
        push!(cs, c)

        p, b1, cost = invert(results(m, b, s, c))
        i = sample(Weights(p))
        b = b1[i]
        total_cost += cost[i]
    end
    push!(bs, b)
    (;total_cost, bs, cs)
end

# rollout(callback::Function, policy; kws...) = rollout(policy; kws..., callback=callback)

