using Parameters
using Distributions
using StatsFuns

"""Notation

All of these are in index space, so 1 means zero.
x: momentary
e1: total evidence for item 1
t1: total time on item 1
"""

@with_kw struct BackwardsInduction
    step_size::Int = 1
    α0::Float64 = 1
    β0::Float64 = 1

    threshold::Int = 10
    sample_cost::Float64 = .001
    switch_cost::Float64 = 0.
    miss_cost::Float64 = 0.
    max_step::Int = 100

    # time and evidence begin at zero but we need indices, so they get shifted +1
    # note e, t, and x are always in index space.
    t_max::Int = max_step + 1
    e_max::Int = threshold + 1

    T::Array{Float64,3} = beta_binomial_pdf(step_size, α0, β0, t_max, e_max)  # x, e, t
    V::Array{Float64,5} = fill(NaN, 2, e_max, e_max, t_max, t_max)  # last_action, e1, e2, t1, t2
    Q::Array{Float64,6} = fill(NaN, 2, 2, e_max, e_max, t_max, t_max)  # action, last_action, e1, e2, t1, t2
end

function beta_binomial_pdf(step_size, α0, β0, t_max, e_max)
    T = zeros(step_size+1, e_max, t_max)   # p(x | e, t)
    for t in 1:t_max
        for e in 1:min(e_max, (t-1)*step_size + 1)
            d = BetaBinomial(step_size, α0 + e-1, β0 + (t-1)*step_size - (e-1))
            for x in 1:step_size+1  # x=1 means zero evidence
                T[x, e, t] = pdf(d, x-1)
            end
        end
    end
    T
end

function Base.show(io::IO, model::BackwardsInduction)
    println(io, "BackwardsInduction")
    for k in [:step_size, :α0, :β0, :sample_cost, :switch_cost, :miss_cost, :max_step]
        println(io, "  $k: ", getfield(model, k))
    end
end

function compute_value_functions!(model::BackwardsInduction)
    @unpack step_size, sample_cost, switch_cost, miss_cost, e_max, t_max, T, V, Q = model

    # shift everything to index space
    # initialize value function in terminal states
    tt = t_max  # tt = total time; start at the end
    for t1 in 1:t_max
        #t1 = t_max
        t2 = (t_max-1) - (t1-1) + 1
        for e2 in 1:min(e_max, (t2-1)*step_size + 1)
            for e1 in 1:min(e_max, (t1-1)*step_size + 1)
                r = e1 == e_max || e2 == e_max ? 0. : -miss_cost
                for last_a in 1:2
                    V[last_a, e1, e2, t1, t2] = r
                end
            end
        end
    end

    # iterate backward in time
    for tt in t_max-1:-1:1
        # iterate over states
        for t1 in 1:tt
            t2 = (tt-1) - (t1-1) + 1
            e2_max = min(e_max, (t2-1)*step_size + 1)
            e1_max = min(e_max, (t1-1)*step_size + 1)
            for e2 in 1:e2_max
                for e1 in 1:e1_max
                    for last_a in 1:2

                        if e1 == e_max || e2 == e_max
                            V[last_a, e1, e2, t1, t2] = 0
                            continue
                        end

                        # sample from 1:  Q(s, a) = sum p(s′|s,a) * V(s′) - cost
                        a = 1
                        cost = (last_a != a || tt == 1) ? sample_cost + switch_cost : sample_cost
                        q1 = -cost
                        for x in 1:step_size+1
                            e1′ = min(e1 + x - 1, e_max)  # -1 b/c index space, min b/c can't exceed threshold
                            #@assert isfinite(V[a, e1′, e2, t1+1, t2])
                            q1 += T[x, e1, t1] * V[a, e1′, e2, t1+1, t2]
                        end
                        Q[a, last_a, e1, e2, t1, t2] = q1
                        
                        # sample from 2
                        a = 2
                        cost = (last_a != a || tt == 1) ? sample_cost + switch_cost : sample_cost
                        q2 = -cost
                        for x in 1:step_size+1
                            e2′ = min(e2 + x - 1, e_max)
                            #@assert isfinite(V[a, e1, e2′, t1, t2+1])
                            q2 += T[x, e2, t2] * V[a, e1, e2′, t1, t2+1]
                        end
                        Q[a, last_a, e1, e2, t1, t2] = q2

                        # terminate
                        #q3 = Q[3, last_a, e1+1, e2+1, t1, tt] = term_reward(model, μ1, μ2)

                        # V(s) = max Q(s, a)
                        V[last_a, e1, e2, t1, t2] = max(q1, q2) # , q3
                    end
                end
            end
        end
    end
end


function BackwardsInduction(m::MetaMDP)
    α0, β0 = m.prior
    b = BackwardsInduction(; m.step_size, α0, β0, m.threshold, m.sample_cost, m.switch_cost, m.miss_cost, m.max_step)
    compute_value_functions!(b)
    b
end

function belief2index(m::MetaMDP, b::Belief)
    e1 = b.heads[1] + 1
    t1 = (b.heads[1] + b.tails[1]) ÷ m.step_size + 1
    e2 = b.heads[2] + 1
    t2 = (b.heads[2] + b.tails[2]) ÷ m.step_size + 1
    (b.focused, e1, e2, t1, t2)
end

function value(B::BackwardsInduction, b::Belief)
    B.V[belief2index(m, b)...]
end


struct OptimalPolicy
    m::MetaMDP
    B::BackwardsInduction
end    

OptimalPolicy(m::MetaMDP) = OptimalPolicy(m, BackwardsInduction(m))

function act(pol::OptimalPolicy, b::Belief)
    f, e1, e2, t1, t2 = belief2index(pol.m, b)
    q = @view pol.B.Q[:, f, e1, e2, t1, t2]
    argmax(q)
end

struct SoftOptimalPolicy
    m::MetaMDP
    B::BackwardsInduction
    β::Float64
end    

SoftOptimalPolicy(m::MetaMDP, β) = SoftOptimalPolicy(m, BackwardsInduction(m), β)

function act(pol::SoftOptimalPolicy, b::Belief)
    f, e1, e2, t1, t2 = belief2index(pol.m, b)
    q = @view pol.B.Q[:, f, e1, e2, t1, t2]
    sample(Weights(softmax(pol.β .* q)))
end
