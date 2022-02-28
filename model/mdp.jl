using Parameters
using Distributions
#using SplitApplyCombine
#using StatsBase
#using StatsFuns

@with_kw struct MetaMDP{N}
    allow_stop::Bool = false
    threshold::Float64 = 1.
    sample_cost::Float64 = 1.
    switch_cost::Float64 = 0.
    miss_cost::Float64 = 0.
    max_step::Int = 100
    noise::Float64 = 1.
    prior::Normal{Float64} = Normal(1., 1.)
end

getfields(x) = (getfield(x, f) for f in fieldnames(typeof(x)))
id(m::MetaMDP) = join(getfields(m), "-")

State{N} = NTuple{N,Float64}
function sample_state(m::MetaMDP{N})::State{N} where N
    Tuple(rand(m.prior, N))
end

mutable struct Belief{N}
    n_step::Int
    focused::Int
    evidence::Vector{Float64}
    time::Vector{Int}
end

function posterior(m::MetaMDP, evidence, time)
    λ_obs = m.noise ^ -2
    λ_prior = m.prior.σ^-2
    μ_prior = m.prior.μ

    λ = λ_obs * time + λ_prior
    μ = (evidence * λ_obs + μ_prior * λ_prior) / λ
    σ = λ^-0.5
    Normal(μ, σ)
end

function posterior(m::MetaMDP, b::Belief)
    posterior.([m], b.evidence, b.time)
end

# function Base.show(io::IO, b::Belief{1})
    # print(io, "($(b.heads[1]), $(b.tails[1])) $(b.n_step)")
# end

# function Base.show(io::IO, b::Belief{2})
#     h1, h2 = b.heads
#     t1, t2 = b.tails
#     print("Belief(")
#     if b.focused == 1
#         print(io, "<($h1, $t1)> ($h2, $t2) $(b.n_step)")
#     elseif b.focused == 2
#         print(io, "($h1, $t1) <($h2, $t2)> $(b.n_step)")
#     else
#         print(io, "($h1, $t1) ($h2, $t2) $(b.n_step)")
#     end
#     print(")")
# end

function initial_belief(m::MetaMDP{N})::Belief{N} where N
    Belief{N}(0, 1, zeros(N), zeros(Int, N))
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

    b.evidence[c] += rand(Normal(s[c], m.noise))
    b.time[c] += 1
    b.n_step += 1
    b.focused = c

    if b.evidence[c] >= m.threshold
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

struct BeliefLog2
    beliefs::Vector{Belief}
end
BeliefLog2() = BeliefLog2(Belief[])

Base.push!(g::BeliefLog2, b) = push!(g.beliefs, deepcopy(b))



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

struct StayPolicy <: Policy
    m::MetaMDP
end

act(pol::StayPolicy, b::Belief{N}) where N = 1

mutable struct RandomStoppingPolicy <: Policy
    m::MetaMDP
    dist::Distribution
    time_to_stop::Int
end

RandomStoppingPolicy(m::MetaMDP{1}, d::Distribution) = RandomStoppingPolicy(m, d, 0)

function act(pol::RandomStoppingPolicy, b::Belief)
    if b.n_step == 1  # initialize
        pol.time_to_stop = ceil(Int, rand(pol.dist))  # note: could be 1 (current step)
    end
    @assert 1 ≤ b.n_step ≤ pol.time_to_stop

    if b.n_step == pol.time_to_stop
        0
    else
        1
    end
end

mutable struct RandomSwitchingPolicy{D<:Distribution} <: Policy 
    m::MetaMDP
    switch_dist::D
    stop_dist::D
    time_to_switch::Int
    time_to_stop::Int
end

function RandomSwitchingPolicy(m::MetaMDP, switch_dist::Distribution, stop_dist::Distribution)
    RandomSwitchingPolicy(m, switch_dist, stop_dist, 0, 0)
end

function act(pol::RandomSwitchingPolicy, b::Belief)
    if b.n_step == 1  # initialize
        pol.time_to_stop = ceil(Int, rand(pol.stop_dist))
        pol.time_to_switch = ceil(Int, rand(pol.switch_dist))
    end
    @assert 1 ≤ b.n_step ≤ min(pol.time_to_stop, pol.time_to_switch)

    if b.n_step == pol.time_to_stop
        0
    else
        if b.n_step == pol.time_to_switch
            pol.time_to_switch = b.n_step + ceil(Int, rand(pol.switch_dist))
            [2, 1][b.focused]
        else
            b.focused
        end
    end
end
