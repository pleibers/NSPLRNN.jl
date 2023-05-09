# NSPLRNN

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pleibers.github.io/NSPLRNN.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pleibers.github.io/NSPLRNN.jl/dev/)
[![Build Status](https://travis-ci.com/pleibers/NSPLRNN.jl.svg?branch=main)](https://travis-ci.com/pleibers/NSPLRNN.jl)
[![Coverage](https://codecov.io/gh/pleibers/NSPLRNN.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pleibers/NSPLRNN.jl)

A package that implements the models presented in the Master thesis by Patrick Leibersperger.

The mlp/linear na/nsPLRNNs. 


## Installation

Either do `add link_to_package` ,manually via the package manager (only allows 1 unregistered package per environment)

Or add the registry at https://gitlab.zi.local/Patrick.Leibersperge/LabRegistry.jl and then use this package as any other registered package.

You might need to remove BPTT and add it manually after with `add link_to_BPTT`

## Usage Information

The time input needs to be the first dimension of the external inputs, with the actual inputs concatenated thereafter.
