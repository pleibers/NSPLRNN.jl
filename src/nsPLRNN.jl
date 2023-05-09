

mutable struct mlpNSPLRNN{V<:AbstractVector,M<:AbstractMatrix,N<:Flux.Chain} <: AbstractNSPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    nn1::N
    nn2::N
    NL₁::M
    NL::M
    C::Union{M,Nothing}
end
@functor mlpNSPLRNN

"""
    mlpNSPLRNN(M::Int, hidden_dim::Int,nl_dim::Int)

clipped formulation of the naPLRNN with mlp as forcing signal, and without external inputs 

as optional model arg:
nl_dim: The dimension the mlp projects the input to (t->2*nl_dim->nl_dim, 2 layer mlp)
"""
function mlpNSPLRNN(M::Int, hidden_dim::Int, nl_dim::Int)
    K = 1
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = uniform_init((hidden_dim,))
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    nn1 = Flux.Chain(
        Flux.Dense(K => 2 * nl_dim, tanh),
        Flux.Dense(2 * nl_dim => nl_dim, tanh)
    )
    nn2 = Flux.Chain(
        Flux.Dense(K => 2 * nl_dim, tanh),
        Flux.Dense(2 * nl_dim => nl_dim, tanh)
    )
    C₂ = uniform_init((hidden_dim, nl_dim))
    C₁ = uniform_init((M, nl_dim))
    return mlpNSPLRNN(A, W₁, W₂, h₁, h₂, nn1, nn2, C₁, C₂, nothing)
end

"""
    mlpNSPLRNN(M::Int, hidden_dim::Int,nl_dim::Int, K::Int)

clipped formulation of the nsPLRNN with mlp as forcing signal, and with external inputs 
of dimension K

as optional model arg:
nl_dim: The dimension the mlp projects the input to (t->2*nl_dim->nl_dim, 2 layer mlp)
"""
function mlpNSPLRNN(M::Int, hidden_dim::Int, nl_dim::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = uniform_init((hidden_dim,))
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    nn1 = Flux.Chain(
        Flux.Dense(1 => 2 * nl_dim, tanh),
        Flux.Dense(2 * nl_dim => nl_dim, tanh)
    )
    nn2 = Flux.Chain(
        Flux.Dense(1 => 2 * nl_dim, tanh),
        Flux.Dense(2 * nl_dim => nl_dim, tanh)
    )
    C₂ = uniform_init((hidden_dim, nl_dim))
    C₁ = uniform_init((M, nl_dim))
    C = uniform_init((M, K))
    return mlpNSPLRNN(A, W₁, W₂, h₁, h₂, nn1, nn2, C₁, C₂, C)
end


"""
    BPTT.PLRNNs.step(m::deltaPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat)

Evolve `z` in time for one step according to the model `m` (equation).

External Inputs are used inside of the non linearity

`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.

"""
function BPTT.PLRNNs.step(m::mlpNSPLRNN, z::AbstractVecOrMat, s::AbstractVector)
    return m.A .* z .+ (m.W₁ .+ m.C₁ * m.nn1(s)) * (relu.((m.W₂ .+ m.C * m.nn2(s)) * z .+ m.h₂) - relu.((m.W₂ .+ m.C * m.nn2(s)) * z)) .+ m.h₁
end
function BPTT.PLRNNs.step(m::mlpNSPLRNN, z::AbstractVecOrMat, time::AbstractMatrix)
    z = m.(eachcol(z), eachcol(time))
    return reduce(hcat, z)
end

mutable struct linearNSPLRNN{V<:AbstractVector,M<:AbstractMatrix} <: AbstractNSPLRNN
    A::V
    W₁₀::M
    W₁₁::M
    W₂₀::M
    W₂₁::M
    h₁::V
    h₂::V
    C::Union{M,Nothing}
end
@functor linearNSPLRNN

"""
    linearNSPLRNN(M::Int, hidden_dim::Int)

clipped formulation of the naPLRNN with linear forcing signal, 
and without external inputs 
"""
function linearNSPLRNN(M::Int, hidden_dim::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = uniform_init((hidden_dim,))
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    W1, W2 = initialize_Ws(M, hidden_dim)
    return linearNSPLRNN(A, W₁, W1, W₂, W2, h₁, h₂, nothing)
end

"""
    linearNSPLRNN(M::Int, hidden_dim::Int, K::Int)

clipped formulation of the naPLRNN with linear forcing signal, 
and with external inputs of dimension K
"""
function linearNSPLRNN(M::Int, hidden_dim::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = uniform_init((hidden_dim,))
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    W1, W2 = initialize_Ws(M, hidden_dim)
    C = uniform_init((M, K))
    return linearNSPLRNN(A, W₁, W1, W₂, W2, h₁, h₂, C)
end

"""
    BPTT.PLRNNs.step(m::linearNSPLRNN, z::AbstractVecOrMat, s::AbstractVector)

Evolve `z` in time for one step according to the model `m` (equation).

External Inputs are used inside of the non linearity

`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.

"""
function BPTT.PLRNNs.step(m::linearNSPLRNN, z::AbstractVecOrMat, s::AbstractVector)
    return m.A .* z .+ (m.W₁₀ .+ m.W₁₁ .* s) * (relu.((m.W₂₀ .+ m.W₂₁ .* s) * z .+ m.h₂) - relu.((m.W₂₀ .+ m.W₂₁ .* s) * z)) .+ m.h₁
end
function BPTT.PLRNNs.step(m::linearNSPLRNN, z::AbstractVecOrMat, time::AbstractMatrix)
    z = m.(eachcol(z), eachcol(time))
    return reduce(hcat, z)
end