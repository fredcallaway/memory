using Parameters
using Distributions
#using SplitApplyCombine
#using StatsBase
#using StatsFuns

@with_kw struct MetaMDP{N}
    threshold::Int = 100
    sample_cost::Float64 = 1.
    switch_cost::Float64 = 0.
    miss_cost::Float64 = 0.
    step_size::Int = 4
    max_step::Int = 100
    allow_stop::Bool = false
    prior::Tuple{Float64, Float64} = (1, 1)
end

getfields(x) = (getfield(x, f) for f in fieldnames(typeof(x)))
id(m::MetaMDP) = join(getfields(m), "-")

State{N} = NTuple{N,Float64}
function sample_state(m::MetaMDP{N})::State{N} where N
    Tuple(rand(Beta(m.prior...), N))
end

mutable struct Belief{N}
    n_step::Int
    focused::Int
    heads::Vector{Int}
    tails::Vector{Int}
end

function Base.show(io::IO, b::Belief{1})
    print(io, "($(b.heads[1]), $(b.tails[1])) $(b.n_step)")
end

function Base.show(io::IO, b::Belief{2})
    h1, h2 = b.heads
    t1, t2 = b.tails
    print("Belief(")
    if b.focused == 1
        print(io, "<($h1, $t1)> ($h2, $t2) $(b.n_step)")
    elseif b.focused == 2
        print(io, "($h1, $t1) <($h2, $t2)> $(b.n_step)")
    else
        print(io, "($h1, $t1) ($h2, $t2) $(b.n_step)")
    end
    print(")")
end

function initial_belief(m::MetaMDP{N})::Belief{N} where N
    Belief{N}(0, 1, zeros(Int, N), zeros(Int, N))
end
is_terminal(b) = b.n_step == -1

function step!(m::MetaMDP, s::State, b::Belief, c::Int)
    #@assert !is_terminal(b)
    #@assert b.n_step < m.max_step
    if c == 0  # give up
        @assert m.allow_stop
        b.n_step = -1
        b.focused = -1
        return m.miss_cost
    end
    cost = (b.n_step == 0 || c != b.focused) ? m.sample_cost + m.switch_cost : m.sample_cost

    # marginalized version 
    #heads = rand(BetaBinomial(m.step_size, m.prior[1] + b.heads[c], m.prior[2] + b.tails[c]))
    heads = rand(Binomial(m.step_size, s[c]))

    tails = m.step_size - heads
    b.heads[c] += heads
    b.tails[c] += tails
    b.focused = c
    b.n_step += 1

    if b.heads[c] >= m.threshold
        b.n_step = -1  # done!
    elseif b.n_step == m.max_step # time out
        cost += m.miss_cost
        b.n_step = -1
        b.focused = -1  # mark failure
    end

    return cost
end

abstract type Policy end

struct NoLog end
Base.push!(::NoLog, b) = nothing

mutable struct NumFixLog
    focused::Int
    n::Int
end
NumFixLog() = NumFixLog(0, 0)
function Base.push!(g::NumFixLog, c)
    if c != g.focused
        g.focused = c
        g.n +=1 
    end
end

mutable struct FullFixLog
    focused::Int
    fixations::Vector{Int}
end
FullFixLog() = FullFixLog(0, [])
function Base.push!(g::FullFixLog, c)
    if c != g.focused
        g.focused = c
        push!(g.fixations, 1)
    else
        g.fixations[end] += 1
    end
end

mutable struct RTLog
    rt::Int
end
RTLog() = RTLog(0)
function Base.push!(g::RTLog, c)
    g.rt += 1
end

function simulate(policy; b=initial_belief(policy.m), s::State=sample_state(policy.m),
                 belief_log=NoLog(), fix_log=NumFixLog())
    m = policy.m
    total_cost = 0.
    c = 1  # first fixation is always to first item
    while true
        push!(belief_log, b)
        push!(fix_log, c)
        total_cost += step!(m, s, b, c)
        is_terminal(b) && break
        c = act(policy, b)
    end
    push!(belief_log, b)
    (;total_cost, b, s, belief_log, fix_log)
end

# %% ==================== Policies ====================

struct RandomPolicy <: Policy
    m::MetaMDP
end

act(pol::RandomPolicy, b::Belief) = rand(1:2)


struct RandomSwitchPolicy <: Policy
    m::MetaMDP
    p_switch::Float64
end

function act(pol::RandomSwitchPolicy, b::Belief)
    if b.focused == 0 || rand() < pol.p_switch
        rand(setdiff(1:2, b.focused))
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
        rand(setdiff(1:2, b.focused))
    else
        pol.time_to_switch -= 1
        b.focused
    end
end


#struct OptimalPolicy <: Policy
#    m::MetaMDP
#    V::ValueFunction
#end

#OptimalPolicy(m::MetaMDP) = OptimalPolicy(m, ValueFunction(m))
#OptimalPolicy(V::ValueFunction) = OptimalPolicy(V.m, V)

#function act(pol::OptimalPolicy, b::Belief)
#    rand(argmaxes(c -> Q(pol.V, b, c), actions(pol.m, b)))
#end


#struct SoftOptimalPolicy <: Policy
#    m::MetaMDP
#    V::ValueFunction
#    β::Float64
#end

#SoftOptimalPolicy(m::MetaMDP; β::Real) = SoftOptimalPolicy(m, ValueFunction(m), float(β))
#SoftOptimalPolicy(V::ValueFunction; β::Real) = SoftOptimalPolicy(V.m, V, float(β))

#function act(pol::SoftOptimalPolicy, b::Belief)
#    p = softmax!(map(c-> pol.β * Q(pol.V, b, c), actions(pol.m, b)))
#    sample(actions(pol.m, b), Weights(p))
#end


# rollout(callback::Function, policy; kws...) = rollout(policy; kws..., callback=callback)

