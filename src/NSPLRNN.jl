# SPDX-License-Identifier: MIT
module NSPLRNN

using BPTT

abstract type AbstractNAPLRNN <: BPTT.AbstractShallowPLRNN end
abstract type AbstractNSPLRNN <: AbstractNAPLRNN end

include("naPLRNN.jl")
include("nsPLRNN.jl")

export mlpNAPLRNN, linearNAPLRNN, clNAPLRNN
export mlpNSPLRNN, linearNSPLRNN


end
