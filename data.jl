module Data
using MLDatasets, Random, LinearAlgebra

export get_mnist_loader

function onehot(label::Int, num_classes::Int=10)
    output = zeros(Float32, 1, num_classes)
    output[label + 1] = 1.0f0
    return output
end

function get_mnist_loader(; batch_size::Int=64, split::Symbol=:train)
    mnist_data = MLDatasets.MNIST(split=split)
    flat_x = Float32.(reshape(mnist_data.features, 28^2, :))
    onehot_y = vcat([onehot(y) for y in mnist_data.targets]...)'
    num_samples = size(flat_x, 2)
    indices = shuffle(1:num_samples)

    batches = []
    for i in 1:batch_size:num_samples
        batch_indices = indices[i:min(i + batch_size - 1, num_samples)]
        
        batch_x = permutedims(flat_x[:, batch_indices])
        batch_y = permutedims(onehot_y[:, batch_indices])
        
        push!(batches, (batch_x, batch_y))
    end
    
    return batches
end

end