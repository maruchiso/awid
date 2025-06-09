module MLP

using ..AutoDiff
using Random, LinearAlgebra

export Dense, Model, get_params

struct Dense
    W::Node
    b::Node
    activation::Function

    function Dense(input_size::Int, output_size::Int, activation_func::Function=identity; dtype::Type{<:Real}=Float32)
        # Xavier/Glorot initialization for weights
        limit = sqrt(dtype(6.0) / (dtype(input_size) + dtype(output_size)))
        W_val = rand(dtype, input_size, output_size) .* dtype(2.0) .* limit .- limit
        b_val = zeros(dtype, 1, output_size)

        W_node = Node(W_val; is_trainable=true)
        b_node = Node(b_val; is_trainable=true)
        
        new(W_node, b_node, activation_func)
    end
end

function forward(layer::Dense, x::Node)
    # x has shape (batch_size, input_features)
    # layer.W has shape (input_features, output_features)
    linear_combination = matmul(x, layer.W) + layer.b
    return layer.activation(linear_combination)
end

function get_params(layer::Dense)
    return [layer.W, layer.b]
end


struct Model
    layers::Vector{Dense}
    parameters::Vector{Node}

    function Model(model_layers::Dense...)
        layers_vec = collect(Dense, model_layers)
        
        all_params = Node[]
        for l in layers_vec
            append!(all_params, get_params(l))
        end
        new(layers_vec, all_params)
    end
end

# Forward pass through the model
function forward(model::Model, x::Node)
    current_output = x
    for layer_item in model.layers
        current_output = forward(layer_item, current_output)
    end
    return current_output
end

# Call Model as function
(model::Model)(x::Node) = forward(model, x)

function get_params(model::Model)
    return model.parameters
end

end