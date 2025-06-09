module AutoDiff

using Statistics, Random, LinearAlgebra

export Node, value, grad, grad!, backward!, zero_grad!, parameters, matmul, relu, sigmoid, softmax, log

include("node_core.jl")    
include("gradients.jl")
include("operators.jl")    
end