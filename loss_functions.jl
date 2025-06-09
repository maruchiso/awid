module LossFunctions

using ..AutoDiff 

export binary_cross_entropy, crossentropy

function binary_cross_entropy(y_pred::Node{T}, y_true::Real) where {T<:Real}
    epsilon = T(1e-8)
    y_true_T = T(y_true)
    
    # Binary Cross Entropy =  - (y * log(p) + (1-y) * log(1-p))
    # half1 = y * log(p)
    half1 = y_true_T * AutoDiff.log(y_pred; epsilon=epsilon)
    
    # half2 = (1-y) * log(1-p)
    one_node = Node(one(T); is_trainable=false)
    half2 = (one(T) - y_true_T) * AutoDiff.log(one_node - y_pred; epsilon=epsilon)
    
    loss_node = -(half1 + half2)

    return AutoDiff.sum(loss_node)
end

function crossentropy(y_pred_probs::Node{T}, y_true_onehot) where T
    y_true_node = Node(y_true_onehot; is_trainable=false)
    
    log_probs = AutoDiff.log(y_pred_probs; epsilon=T(1e-9))
    
    total_loss = AutoDiff.sum(-(y_true_node * log_probs))
    
    batch_size = size(value(y_pred_probs), 1)
    return total_loss / Node(T(batch_size); is_trainable=false)
end

end 