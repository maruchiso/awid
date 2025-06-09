include("./autodiff/autodiff.jl")
include("./loss_functions.jl")
include("./mlp.jl")
include("./optimizer.jl")
include("./data.jl") 

using .AutoDiff, .LossFunctions, .MLP, .Optimizers, .Data
using Printf, Random, Statistics, JSON


const LEARNING_RATE = Float32(3e-4)
const EPOCHS = 30
const BATCH_SIZE = 256
const PARAM_DTYPE = Float32

const INPUT_SIZE = 28^2           # 28x28 pixels
const HIDDEN_SIZE = 32            # 32 neurons
const OUTPUT_SIZE = 10            # 10 neuronów (0 - 9 digits)

function main()
    Random.seed!(42)

    println("Load data")
    train_loader = Data.get_mnist_loader(batch_size=BATCH_SIZE, split=:train)
    test_loader = Data.get_mnist_loader(batch_size=1024, split=:test)

    model = MLP.Model(
        # Hidden layer with relu activation
        MLP.Dense(INPUT_SIZE, HIDDEN_SIZE, AutoDiff.relu; dtype=PARAM_DTYPE),
        # Output Layer
        MLP.Dense(HIDDEN_SIZE, OUTPUT_SIZE, identity; dtype=PARAM_DTYPE)
    )
    params = MLP.get_params(model)
    typed_params = Vector{Node{PARAM_DTYPE}}(params)
    println("MLP: 28x28 -> $(HIDDEN_SIZE) (relu) -> $(OUTPUT_SIZE)")

    # Loss function and optimizer
    loss_fn = LossFunctions.crossentropy
    optimizer = Optimizers.Adam(typed_params; lr=LEARNING_RATE)

    function accuracy(model_to_test, data_loader)
        correct_predictions = 0
        total_samples = 0
        for (x, y_onehot) in data_loader
            logits = model_to_test(Node(x))
            # Softmax for probabilities
            probs = AutoDiff.softmax(logits)
            
            # Find the predicted label
            preds = getindex.(findmax(value(probs), dims=2)[2], 2) .- 1
            true_labels = getindex.(findmax(y_onehot, dims=2)[2], 2) .- 1
            
            correct_predictions += sum(preds .== true_labels)
            total_samples += size(x, 1)
        end

        return round(100 * correct_predictions / total_samples; digits=2)
    end

    global_start_time = time()
    global_avg_loss = zero(PARAM_DTYPE)
    for epoch in 1:EPOCHS
        total_loss_epoch = zero(PARAM_DTYPE)
        for (x_batch, y_batch) in train_loader
            # Forward pass
            logits = model(Node(x_batch; is_trainable=false))
            probs = AutoDiff.softmax(logits)
            loss = loss_fn(probs, y_batch)
            total_loss_epoch += value(loss)[1]

            # Backward pass and optimization
            AutoDiff.zero_grad!(params)
            AutoDiff.backward!(loss)
            Optimizers.update!(optimizer)
        end
        
        avg_loss = total_loss_epoch / length(train_loader)

        test_acc = accuracy(model, test_loader)
        @printf("Epoch: %2d / %2d, Avg loss: %.6f, accuracy (Test): %.2f%%\n", epoch, EPOCHS, avg_loss, test_acc)
        global_avg_loss = avg_loss
    end

    global_end_time = time()
    total_train_time = global_end_time - global_start_time
    final_test_accuracy = accuracy(model, test_loader)

    results = Dict(
        "loss" => global_avg_loss,
        "accuracy" => final_test_accuracy,
        "train_time" => total_train_time
    )

    open("mlp.json", "w") do f
        JSON.print(f, results)
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end