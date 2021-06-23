using JSON
using DataFrames
using Memoize

@memoize function load_fixations()
    multi = mapreduce(vcat, VERSIONS) do version
        CSV.read("../data/$version/multi-recall.csv", DataFrame)
    end
    filter!(multi) do row
        !row.practice
    end
    # TODO: throw out bad trials - not sure what I meant...
    Vector{Vector{Float64}}(map(JSON.parse, multi.presentation_times))
end

function empirical_policy(m)
    fixations = load_fixations()
    ms_per_sample = 1000 * (TIME_LIMIT / m.max_step)
    x = round.(Int, reduce(vcat, fixations) ./ ms_per_sample)
    x = max.(x, 1)
    d = fit(DiscreteNonParametric, x)
    SwitchDistributionPolicy(m, d)
end

function get_last_other(df)
    last_ = Int[]; other = Int[]
    for pt in df.presentation_times
        if !isempty(pt)
            push!(last_, pt[end])
            push!(other, pt[1:end-1]...)
        end
    end
    mean(last_), mean(other)
end

mutable struct RandomCommitmentPolicy{D<:Distribution} <: Policy 
    m::MetaMDP
    dist::D
    time_to_switch::Int
    p_commit::Float64
end

function act(pol::RandomCommitmentPolicy, b::Belief)
    if b.n_step == 0 || pol.time_to_switch == 0
        commit = rand() < pol.p_commit
        pol.time_to_switch = commit ? -1 : ceil(Int, rand(pol.dist)) - 1
        rand(setdiff(1:n_item(b), b.focused))
    else
        pol.time_to_switch -= 1
        b.focused
    end
end

function empirical_p_commit()
    fixations = load_fixations()
    n_fix = filter!(x->x>0, length.(fixations))
    d = fit(Geometric, n_fix .- 1)
    d.p
end

function empirical_commitment_policy(m, p_commit=empirical_p_commit())
    fixations = load_fixations()
    ms_per_sample = 1000 * (TIME_LIMIT / m.max_step)
    non_final = mapreduce(vcat, fixations) do fixtimes
        max.(1, round.(Int, fixtimes[1:end-1] ./ ms_per_sample))
    end
    d = fit(DiscreteNonParametric, non_final)
    RandomCommitmentPolicy(m, d)
end

# %% --------

mutable struct RandomCommitmentPolicy2{D<:Distribution,E<:Distribution} <: Policy 
    m::MetaMDP
    duration_dist::D
    commitment_dist::E
    time_to_switch::Int
    time_to_commit::Int
end

function act(pol::RandomCommitmentPolicy2, b::Belief)
    if b.n_step == 0
        pol.time_to_commit = rand(pol.commitment_dist)
        @assert pol.time_to_commit > 0
    end
    if b.n_step == 0 || pol.time_to_switch == 0
        pol.time_to_commit -= 1
        commit = pol.time_to_commit == 0
        pol.time_to_switch = commit ? -1 : ceil(Int, rand(pol.duration_dist)) - 1
        #@assert pol.time_to_switch >= 0
        rand(setdiff(1:n_item(b), b.focused))
    else
        pol.time_to_switch -= 1
        b.focused
    end
end

function empirical_commitment_policy2(m)
    fixations = load_fixations()
    n_fix = filter!(x->x>0, length.(fixations))
    commitment_dist = fit(DiscreteNonParametric, n_fix)

    ms_per_sample = 1000 * (TIME_LIMIT / m.max_step)
    non_final = mapreduce(vcat, fixations) do fixtimes
        max.(1, round.(Int, fixtimes[1:end-1] ./ ms_per_sample))
    end
    duration_dist = fit(DiscreteNonParametric, non_final)

    RandomCommitmentPolicy2(m, duration_dist, commitment_dist, 0, 0)
end

pol = empirical_commitment_policy2(m)

# %% --------

figure() do
    fixations = load_fixations()
    n_fix = filter!(x->x>0, length.(fixations))  
    histogram(n_fix, bins=-0.5:1:10.5, normalize=:probability)
    #plot!(Geometric(empirical_p_commit()))
    plot!(pol.commitment_dist)
end
