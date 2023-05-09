# SPDX-License-Identifier: MIT
module NSPLRNN

using BPTT
using Flux

abstract type AbstractNAPLRNN <: BPTT.AbstractShallowPLRNN end
abstract type AbstractNSPLRNN <: AbstractNAPLRNN end

include("initialization.jl")
include("naPLRNN.jl")
include("nsPLRNN.jl")

export mlpNAPLRNN, linearNAPLRNN, clNAPLRNN
export mlpNSPLRNN, linearNSPLRNN


end
