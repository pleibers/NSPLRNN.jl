
# shallowPLRNN with t in the non linearity
mutable struct linearNAPLRNN{V<:AbstractVector,M<:AbstractMatrix} <: AbstractNAPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    NL::M
    C::Union{M,Nothing}
end
@functor linearNAPLRNN

# initialization/constructor
"""
    linearNAPLRNN(M::Int, hidden_dim::Int)

naPLRNN with linear forcing signal, and without external inputs

"""
function linearNAPLRNN(M::Int, hidden_dim::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    NL = uniform_init((hidden_dim, 1))
    return linearNAPLRNN(A, W₁, W₂, h₁, h₂, NL, nothing)
end

"""
    linearNAPLRNN(M::Int, hidden_dim::Int,K::Int)

naPLRNN with linear forcing signal, and with external inputs of dimension K


"""
function linearNAPLRNN(M::Int, hidden_dim::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    NL = uniform_init((hidden_dim, 1))
    C = uniform_init((M, K))
    return linearNAPLRNN(A, W₁, W₂, h₁, h₂, NL, C)
end

"""
    BPTT.PLRNNs.step(m::linearNAPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat)

Evolve `z` in time for one step according to the model `m` (equation).

External Inputs are used inside of the non linearity

`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.

"""
function BPTT.PLRNNs.step(m::linearNAPLRNN, z::AbstractVecOrMat, t::AbstractVecOrMat)
    return m.A .* z .+ m.W₁ * relu.(m.W₂ * z .+ m.h₂ .+ m.NL * t) .+ m.h₁
end


mutable struct clNAPLRNN{V<:AbstractVector,M<:AbstractMatrix} <: BPTT.AbstractShallowPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    NL::M
    C::Union{M,Nothing}
end
@functor clNAPLRNN (A, W₁, W₂, h₁, h₂, C)

# initialization/constructor
"""
    clNAPLRNN(M::Int, hidden_dim::Int)

clipped formulation of the naPLRNN with linear forcing signal, and without external inputs

"""
function clNAPLRNN(M::Int, hidden_dim::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = uniform_init((hidden_dim,))
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    NL = uniform_init((hidden_dim, 1))

    return clNAPLRNN(A, W₁, W₂, h₁, h₂, NL, nothing)
end

"""
    clNAPLRNN(M::Int, hidden_dim::Int, K::Int)

clipped formulation of the naPLRNN with linear forcing signal, 
and with external inputs of dimension K
"""
function clNAPLRNN(M::Int, hidden_dim::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = uniform_init((hidden_dim,))
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    C = uniform_init((M, K))
    NL = uniform_init((hidden_dim, 1))

    return clNAPLRNN(A, W₁, W₂, h₁, h₂, NL, C)
end

"""
    BPTT.PLRNNs.step(m::nltPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat)

Evolve `z` in time for one step according to the model `m` (equation).

External Inputs are used inside of the non linearity

`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.

"""
function BPTT.PLRNNs.step(m::clNAPLRNN, z::AbstractVecOrMat, t::AbstractVecOrMat)
    return m.A .* z .+ m.W₁ * (relu.(m.W₂ * z .+ m.h₂ .+ m.NL * t) - relu.(m.W₂ * z)) .+ m.h₁
end


mutable struct mlpNAPLRNN{V<:AbstractVector,M<:AbstractMatrix,N<:Flux.Chain} <: BPTT.AbstractShallowPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    nn::N
    NL::M
    C::Union{M,Nothing}
end
@functor mlpNAPLRNN

"""
    mlpNAPLRNN(M::Int, hidden_dim::Int, nl_dim::Int)

clipped formulation of the naPLRNN with mlp as forcing signal, and without external inputs

as optional model arg:
nl_dim: The dimension the mlp projects the input to (t->2*nl_dim->nl_dim, 2 layer mlp)

"""
function mlpNAPLRNN(M::Int, hidden_dim::Int, nl_dim::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = uniform_init((hidden_dim,))
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    nn = Flux.Chain(
        Flux.Dense(K => 2 * nl_dim, tanh),
        Flux.Dense(2 * nl_dim => nl_dim, tanh)
    )
    NL = uniform_init((hidden_dim, 1))

    return mlpNAPLRNN(A, W₁, W₂, h₁, h₂, nn, NL, nothing)
end

"""
    mlpNAPLRNN(M::Int, hidden_dim::Int, nl_dim::Int, K::Int)

clipped formulation of the naPLRNN with mlp as forcing signal, and with external inputs 
of dimension K

as optional model arg:
nl_dim: The dimension the mlp projects the input to (t->2*nl_dim->nl_dim, 2 layer mlp)
"""
function mlpNAPLRNN(M::Int, hidden_dim::Int, nl_dim::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = uniform_init((hidden_dim,))
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    nn = Flux.Chain(
        Flux.Dense(K => 2 * nl_dim, tanh),
        Flux.Dense(2 * nl_dim => nl_dim, tanh)
    )
    C = uniform_init((M, K))
    NL = uniform_init((hidden_dim, 1))

    return mlpNAPLRNN(A, W₁, W₂, h₁, h₂, nn, NL, C)
end



"""
    BPTT.PLRNNs.step(m::mlpNAPLRNN, z::AbstractVecOrMat, t::AbstractVecOrMat)

Evolve `z` in time for one step according to the model `m` (equation).

`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.

"""
function BPTT.PLRNNs.step(m::mlpNAPLRNN, z::AbstractVecOrMat, t::AbstractVecOrMat)
    return m.A .* z .+ m.W₁ * (relu.(m.W₂ * z .+ m.h₂ .+ m.C * m.nn(t)) - relu.(m.W₂ * z)) .+ m.h₁
end



"""
    BPTT.PLRRNs.step(m::AbstractNAPLRNN, z, s::AbstractMatrix)

Split the time and extarnal inputs for the evolution of z
"""
function BPTT.PLRRNs.step(m::AbstractNAPLRNN, z, s::AbstractMatrix)
    z_t = @views BPTT.PLRNNs.step(m, z, s[1, :])
    if size(s, 1) > 1
        z_t = @views z_t + m.C * s[2:end]
    end
    return z_t
end

"""
    BPTT.PLRRNs.step(m::AbstractNAPLRNN, z, s::AbstractVector)

Split the time and extarnal inputs for the evolution of z

"""
function BPTT.PLRRNs.step(m::AbstractNAPLRNN, z, s::AbstractVector)
    z_t = @views BPTT.PLRNNs.step(m, z, [s[1]])
    if size(s, 1) > 1
        z_t = @views z_t + m.C * s[2:end]
    end
    return z_t
end

