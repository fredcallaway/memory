using Parameters
using Distributions
using StatsFuns
using StatsBase

"""Notation

All of these are in index space, so 1 means zero.
x: momentary
e1: total evidence for item 1
t1: total time on item 1
"""

struct BackwardsInduction{N,N2,N3}
    m::MetaMDP{N}
    dv::Float64

    T::Array{Float64,3}
    V::Array{Float64,N2}  # 1+2N
    Q::Array{Float64,N3}  # 2+2N
end

function BackwardsInduction(m::MetaMDP{N}; dv::Float64=.02m.threshold, compute=true, verbose=false) where {N}
    verbose && println("Computing transition matrix")
    T = transition_matrix(m, dv)
    @assert round(m.threshold / dv) ≈ m.threshold / dv
    e_max = round(Int, 2m.threshold / dv + 1)
    t_max = m.max_step + 1
    sz = if N == 1
        (e_max, t_max)
    elseif N == 2
        (e_max, e_max, t_max, t_max)
    else
        error("Not implemented")
    end
    verbose && println("Allocating value functions")
    V = fill(NaN, N, sz...)  # last_action, e1, (e2), t1, (t2)
    Q = fill(NaN, N, N, sz...)  # action, last_action, e1, (e2), t1, (t2)
    b = BackwardsInduction(m, dv, T, V, Q)
    compute && compute_value_functions!(b; verbose)
    b
end

function transition_matrix(m, dv)
    # make sure we don't lose a grid cell to numerical error
    n_grid = round(Int, 1 + 2m.threshold / dv)
    evidences = range(-m.threshold, m.threshold, length=n_grid)
    @assert (evidences[2] - evidences[1]) ≈ dv

    λ_obs = m.noise ^ -2
    λ_prior = m.prior.σ^-2
    μ_prior = m.prior.μ

    e_max = length(evidences)
    t_max = m.max_step + 1
    T = zeros(e_max, e_max, t_max)   # ev′ | ev, t

    for e in 1:e_max, t in 1:t_max
        ev = evidences[e]; time = t - 1

        # posterior over drift rate
        λ = λ_obs * time + λ_prior
        μ = (ev * λ_obs + μ_prior * λ_prior) / λ
        σ = λ^-0.5

        # posterior predictive over next evidence
        predictive = Normal(ev+μ, √(λ^-1 + λ_obs^-1))

        for e′ in 1:e_max
            ev′ = evidences[e′]
            lower = e′ == 1 ? -Inf : ev′ - dv/2
            upper = e′ == e_max ? Inf : ev′ + dv/2
            T[e′, e, t] = cdf(predictive, upper) - cdf(predictive, lower)
        end
    end
    T
end

function Base.show(io::IO, model::BackwardsInduction)
    print(io, "BackwardsInduction for ", model.m)
end

function compute_value_functions!(model::BackwardsInduction{1}; verbose=true)
    @unpack dv, T, V, Q = model
    @unpack allow_stop, max_step, threshold, sample_cost, switch_cost, miss_cost = model.m
    t_max = max_step + 1
    e_max = size(T, 1)
    a = 1 # we only have this for consistency with two-item case

    @assert allow_stop  # only version that makes sense

    # NOTE: the -1 and +1 everywhere are shifting back and forth
    # between index space and time/evidence (which starts at 0)

    # initialize value function in terminal states
    V[a, :, t_max] .= -miss_cost
    V[a, e_max, :] .= 0

    # iterate backward in time
    for t in t_max-1:-1:1
        # iterate over states
        for e1 in 1:e_max-1

            # sample from 1:  Q(s, a) = sum p(s′|s,a) * V(s′) - cost
            q1 = -sample_cost
            if t == 1
                q1 -= switch_cost
            end
            for e′ in 1:e_max
                q1 += T[e′, e1, t] * V[a, e′, t+1]
            end
            Q[a, a, e1, t] = q1
    
            # terminate
            q2 = -miss_cost

            V[a, e1, t] = max(q1, q2)
        end
    end
end

function compute_value_functions!(model::BackwardsInduction{2}; verbose=true)
    verbose && println("Computing value functions")
    @unpack dv, T, V, Q = model
    @unpack allow_stop, max_step, threshold, sample_cost, switch_cost, miss_cost = model.m
    t_max = max_step + 1
    e_max = round(Int, 2threshold / dv + 1)

    stop_value = allow_stop ? -miss_cost : -Inf

    # shift everything to index space
    # initialize value function in terminal states
    tt = t_max  # tt = total time; start at the end
    for t1 in 1:t_max
        #t1 = t_max
        t2 = (t_max-1) - (t1-1) + 1
        V[:, :, :, t1, t2] .= -miss_cost
    end
    V[:, e_max, :, :, :] .= 0.
    V[:, :, e_max, :, :] .= 0.

    progress = Progress(t_max; desc="Backwards induction ", enabled=verbose)
    # iterate backward in time
    for tt in t_max-1:-1:1
        ProgressMeter.next!(progress)
        # iterate over states
        for t1 in 1:tt
            t2 = (tt-1) - (t1-1) + 1
            for e2 in 1:e_max-1
                for e1 in 1:e_max-1
                    for last_a in 1:2

                        # sample from 1:  Q(s, a) = sum p(s′|s,a) * V(s′) - cost
                        a = 1
                        cost = (last_a != a || tt == 1) ? sample_cost + switch_cost : sample_cost
                        q1 = -cost
                        for e1′ in 1:e_max
                            q1 += T[e1′, e1, t1] * V[a, e1′, e2, t1+1, t2]
                        end
                        Q[a, last_a, e1, e2, t1, t2] = q1
                        
                        # sample from 2
                        a = 2
                        cost = (last_a != a || tt == 1) ? sample_cost + switch_cost : sample_cost
                        q2 = -cost
                        for e2′ in 1:e_max
                            q2 += T[e2′, e2, t2] * V[a, e1, e2′, t1, t2+1]
                        end
                        Q[a, last_a, e1, e2, t1, t2] = q2

                        # V(s) = max Q(s, a)
                        V[last_a, e1, e2, t1, t2] = max(q1, q2, stop_value) # , q3
                    end
                end
            end
        end
    end
end

round_evidence(B, e) = clip(round(Int, (e + B.m.threshold) / B.dv + 1), 1, size(B.V, 2)-1)

function belief2index(B::BackwardsInduction{1}, b::Belief{1})
    e1 = round_evidence(B, b.evidence[1])
    t1 = b.time[1]+1
    (b.focused, e1, t1)
end

function belief2index(B::BackwardsInduction{2}, b::Belief{2})
    e1, e2 = map(e->round_evidence(B, e), b.evidence)
    t1, t2 = b.time .+ 1
    (b.focused, e1, e2, t1, t2)
end

function value(B::BackwardsInduction, b::Belief)
    B.V[belief2index(B, b)...]
end

function value(B::BackwardsInduction, b::Belief, f::Int)
    B.Q[f, belief2index(B, b)...]
end


struct OptimalPolicy{N} <: Policy
    m::MetaMDP{N}
    B::BackwardsInduction{N}
end    

OptimalPolicy(B::BackwardsInduction) = OptimalPolicy(B.m, B)
OptimalPolicy(m::MetaMDP, dv::Float64=.02m.threshold) = OptimalPolicy(m, BackwardsInduction(m; dv))


function act(pol::OptimalPolicy, b::Belief)
    qs = @view pol.B.Q[:, belief2index(pol.B, b)...]
    @assert all(isfinite, qs)
    q, a = findmax(qs)
    if pol.m.allow_stop && q < -pol.m.miss_cost
        0
    else
        a
    end
end

struct SoftOptimalPolicy <: Policy
    m::MetaMDP
    B::BackwardsInduction
    β::Float64
end    

SoftOptimalPolicy(m::MetaMDP, β) = SoftOptimalPolicy(m, BackwardsInduction(m), β)
SoftOptimalPolicy(B::BackwardsInduction, β) = SoftOptimalPolicy(B.m, B, β)

function act(pol::SoftOptimalPolicy, b::Belief)
    qs = @view pol.B.Q[:, belief2index(pol.B, b)...]
    @assert all(isfinite, qs)
    if pol.m.allow_stop
        sample(Weights(softmax(pol.β .* [-pol.m.miss_cost; qs]))) - 1
    else
        sample(Weights(softmax(pol.β .* qs)))
    end
end
