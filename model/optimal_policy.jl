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
    T::Array{Float64,3}
    V::Array{Float64,N2}  # 1+2N
    Q::Array{Float64,N3}  # 2+2N
end

function BackwardsInduction(m::MetaMDP{N}) where {N}
    t_max = m.max_step + 1
    e_max = m.threshold + 1
    T = beta_binomial_pdf(m.step_size, m.prior[1], m.prior[2], t_max, e_max)
    sz = if N == 1
        (e_max, t_max)
    elseif N == 2
        (e_max, e_max, t_max, t_max)
    else
        error("Not implemented")
    end
    V = fill(NaN, N, sz...)  # last_action, e1, (e2), t1, (t2)
    Q = fill(NaN, N, N, sz...)  # action, last_action, e1, (e2), t1, (t2)
    b = BackwardsInduction(m, T, V, Q)
    compute_value_functions!(b)
    b
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
    print(io, "BackwardsInduction for ", model.m)
end

function compute_value_functions!(model::BackwardsInduction{1})
    @unpack T, V, Q = model
    @unpack step_size, sample_cost, switch_cost, miss_cost = model.m
    t_max = model.m.max_step + 1
    e_max = model.m.threshold + 1
    last_a = 1 # we only have this for consistency with two-item case

    @assert m.allow_stop  # only version that makes sense

    # NOTE: the -1 and +1 everywhere are shifting back and forth
    # between index space and time/evidence (which starts at 0)

    # initialize value function in terminal states
    for e1 in 1:min(e_max, (t_max-1)*step_size + 1)
        V[last_a, e1, t_max] = e1 == e_max ? 0. : -miss_cost
    end

    # iterate backward in time
    for tt in t_max-1:-1:1
        t1 = tt   # all time must be on first item

        # iterate over states
        e1_max = min(e_max, (t1-1)*step_size + 1)
        for e1 in 1:e1_max
            if e1 == e_max
                V[last_a, e1, t1] = 0
                continue
            end

            # sample from 1:  Q(s, a) = sum p(s′|s,a) * V(s′) - cost
            a = 1
            cost = sample_cost
            q1 = -cost
            for x in 1:step_size+1
                e1′ = min(e1 + x - 1, e_max)  # -1 b/c index space, min b/c can't exceed threshold
                #@assert isfinite(V[a, e1′, e2, t1+1, t2])
                q1 += T[x, e1, t1] * V[a, e1′, t1+1]
            end
            Q[a, last_a, e1, t1] = q1
            
            # terminate
            q2 = -miss_cost

            V[last_a, e1, t1] = max(q1, q2) # , q3
        end
    end
end

function compute_value_functions!(model::BackwardsInduction{2})
    @unpack T, V, Q = model
    @unpack step_size, sample_cost, switch_cost, miss_cost = model.m
    t_max = model.m.max_step + 1
    e_max = model.m.threshold + 1

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

function belief2index(m::MetaMDP{1}, b::Belief{1})
    e1 = b.heads[1] + 1
    t1 = (b.heads[1] + b.tails[1]) ÷ m.step_size + 1
    (b.focused, e1, t1)
end

function belief2index(m::MetaMDP{2}, b::Belief{2})
    e1 = b.heads[1] + 1
    t1 = (b.heads[1] + b.tails[1]) ÷ m.step_size + 1
    e2 = b.heads[2] + 1
    t2 = (b.heads[2] + b.tails[2]) ÷ m.step_size + 1
    (b.focused, e1, e2, t1, t2)
end

function value(B::BackwardsInduction, b::Belief)
    B.V[belief2index(m, b)...]
end

function value(B::BackwardsInduction, b::Belief, f::Int)
    B.Q[f, belief2index(m, b)...]
end


struct OptimalPolicy{N}
    m::MetaMDP{N}
    B::BackwardsInduction{N}
end    

OptimalPolicy(m::MetaMDP) = OptimalPolicy(m, BackwardsInduction(m))

function act(pol::OptimalPolicy, b::Belief)
    qs = @view pol.B.Q[:, belief2index(pol.m, b)...]
    q, a = findmax(qs)
    if pol.m.allow_stop && q < pol.m.miss_cost
        0
    else
        a
    end
end

struct SoftOptimalPolicy
    m::MetaMDP
    B::BackwardsInduction
    β::Float64
end    

SoftOptimalPolicy(m::MetaMDP, β) = SoftOptimalPolicy(m, BackwardsInduction(m), β)

function act(pol::SoftOptimalPolicy, b::Belief)
    qs = @view pol.B.Q[:, belief2index(pol.m, b)...]
    if pol.m.allow_stop
        sample(Weights(softmax(pol.β .* [-pol.m.miss_cost; qs]))) - 1
    else
        sample(Weights(softmax(pol.β .* qs)))
    end
end
