function build_mlp(; n_hidden=32, n_input=1, n_output=3,twice=true)
    if twice
        n_hidden = n_output + Int(round(n_output*0.5))
    end
    return Chain(
        Dense(n_input => n_hidden, relu, init=uniform_init),
        Dense(n_hidden => n_output, init=uniform_init))
end

function initialize_Ws(M, hidden_dim)
    W₁ = uniform_init((M, hidden_dim))
    W₂ = uniform_init((hidden_dim, M))
    return W₁, W₂
end