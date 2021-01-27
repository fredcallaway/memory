using Parameters
using Distributions
using SplitApplyCombine
using StatsBase

@with_kw struct MetaMDP{N}
    threshold::Int = 100
    sample_cost::Float64 = 1.
    switch_cost::Float64 = 0.
    miss_cost::Float64 = 0.
    step_size::Int = 10
    max_step::Int = 100
end
MetaMDP(;kws...) = MetaMDP{2}(;kws...)


struct Belief{N}
    n_step::Int
    focused::Int
    counts::NTuple{N, Tuple{Int, Int}}
end

initial_belief(m::MetaMDP) = Belief(0, 0, Tuple((1,1) for i in 1:n_item(m)))
terminal_belief(m::MetaMDP, c::Int=-1) = Belief(-1, c, Tuple((-1, -1) for i in 1:n_item(m)))
is_terminal(b) = b.n_step == -1
# max_cost(m::MetaMDP) = m.miss_cost

n_item(m::MetaMDP{N}) where N = N
n_item(b::Belief{N}) where N = N

function results(m::MetaMDP{N}, b::Belief{N}, c::Int) where N
    @unpack threshold, step_size, max_step, sample_cost, switch_cost, miss_cost = m
    @unpack n_step, focused, counts = b
    @assert !is_terminal(b)
    @assert n_step < max_step
    # @assert sum(sum.(b.counts)) - n_step * step_size - 2N == 0
    # if sum(sum.(b.counts)) - n_step * step_size - 2N != 0
    #     @error "Bad belief" b c
    #     error()
    # end
    # n_step == max_step && return [(1., terminal_belief(m), miss_cost)]

    cost = c != focused ? sample_cost + switch_cost : sample_cost
    map(0:step_size) do heads
        α, β = counts[c]
        p = pdf(BetaBinomial(step_size, α, β), heads)
        if α + heads >= threshold
            return (p, terminal_belief(m, c), cost)
        elseif n_step == max_step - 1
            return (p, terminal_belief(m), cost + miss_cost)
        else
            new_counts = Base.setindex(counts, (α + heads, β + (step_size - heads)), c)
            return (p, Belief(n_step + 1, c, new_counts), cost)
        end
    end
end
            
# %% ==================== Solution ====================

struct ValueFunction{F}
    m::MetaMDP
    hasher::F
    cache::Dict{UInt64, Float64}
end

function default_hash(b::Belief{2})
    # hash(b.focused, hash(b.n_step, hash(b.counts)))
    # hash(b.focused, hash(b.counts))
    hash(b.counts[1]) << Int(b.focused == 1) +
    hash(b.counts[2]) << Int(b.focused == 2)
end

ValueFunction(m::MetaMDP, hasher::Function) = ValueFunction(m, hasher, Dict{UInt64, Float64}())
ValueFunction(m::MetaMDP) = ValueFunction(m, default_hash)


function Q(V::ValueFunction, b::Belief, c)::Float64
    sum(p * (r + V(b1)) for (p, b1, r) in results(V.m, b, c))
end

Q(V::ValueFunction, b::Belief) = [Q(V,b,c) for c in 1:n_item(b)]


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
    minimum(Q(V, b, c) for c in 1:n_item(b))
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
    findall(isequal(mfx), fx)    
end

abstract type Policy end

function act(pol::Policy, b::Belief)
    rand(actions(pol, b))
end

struct OptimalPolicy <: Policy
    m::MetaMDP
    V::ValueFunction
end

OptimalPolicy(m::MetaMDP) = OptimalPolicy(m, ValueFunction(m))
OptimalPolicy(V::ValueFunction) = OptimalPolicy(V.m, V)
(pol::OptimalPolicy)(b::Belief) = act(pol, b)

function actions(pol::OptimalPolicy, b::Belief)
    argmaxes(c->Q(pol.V, b, c), 1:n_item(b))
end

function simulate(policy; b=initial_belief(policy.m))
    m = policy.m
    total_cost = 0.
    bs = Belief[]
    cs = Int[]
    while !is_terminal(b)
        c = policy(b)
        push!(bs, b); push!(cs, c)
        p, b1, cost = invert(results(m, b, c))
        i = sample(Weights(p))
        b = b1[i]
        total_cost += cost[i]
    end
    push!(bs, b)
    (;total_cost, bs, cs)
end

# rollout(callback::Function, policy; kws...) = rollout(policy; kws..., callback=callback)

