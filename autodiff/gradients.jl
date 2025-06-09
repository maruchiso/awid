function accumulate_gradient!(node::Node{T}, gradient_update) where {T<:Real}
    g_converted = if isa(gradient_update, AbstractArray)
        convert(AbstractArray{T}, gradient_update)
    elseif isa(gradient_update, Real)
        convert(T, gradient_update)
    else
        @error "Unsupported gradient type for accumulation for Node $(node). Got $(typeof(gradient_update)), expected Real or AbstractArray."
        return
    end

    if node.gradient === nothing
        node.gradient = deepcopy(g_converted)
    else
        existing_grad = node.gradient
        update_grad = g_converted

        if isa(existing_grad, Real)
            if isa(update_grad, Real)
                node.gradient += update_grad
            else
                node.gradient += sum(update_grad)
            end
        else # existing_grad is an array
            if isa(update_grad, Real)
                existing_grad .+= update_grad
            else # update_grad is also an array
                try
                    existing_grad .+= update_grad
                catch e
                    error("Failed to accumulate gradients with incompatible shapes for Node $(node). Existing: $(size(existing_grad)), Update: $(size(update_grad)). Error: $e")
                end
            end
        end
    end
end

function backward!(output_loss_node::Node{T}) where {T<:Real}
    # Initialize gradient
    if output_loss_node.gradient === nothing
        if isa(output_loss_node.value, Real) || length(output_loss_node.value) == 1
            grad!(output_loss_node, one(T))
        else
            error("Backward pass must start from a scalar Node. Got Node with value shape $(size(output_loss_node.value)). Use sum() to reduce loss to a scalar if needed.")
        end
    elseif !(isa(output_loss_node.gradient, Real) || length(output_loss_node.gradient) == 1)
        @warn "Backward pass started on a Node with a non-scalar gradient. Shape: $(size(output_loss_node.gradient))."
    end

    # List with nodes from inputs to outputs
    topological_order = Node[]
    visited = Set{Node}()

    function build_topological_order(current_node::Node)
        push!(visited, current_node)
        # Recurently visit all inputs
        for input_node in current_node.inputs
            if !(input_node in visited)
                build_topological_order(input_node)
            end
        end
        # Add current node to list after all inputs
        push!(topological_order, current_node)
    end

    # Backpropagation
    build_topological_order(output_loss_node)
    for node in reverse(topological_order)
        if node.gradient !== nothing
            node.backward_function()
        else
            @debug "Node $(node) has no gradient set, skipping backward pass."
        end
    end
end

# Zero grad has to be called before each training step to reset gradients.
function zero_grad!(trainable_params::AbstractVector{<:Node})
    if isempty(trainable_params)
        @warn "Tried to zero gradients for an empty list of parameters. No action taken."
        return
    end

    for parameter_node in trainable_params
        if !parameter_node.is_trainable_param
            @warn "Zero grad for Node $(parameter_node) that is not a trainable parameter."
        end
        
        if parameter_node.gradient !== nothing
            el_type = _eltype(parameter_node.value)
            if isa(parameter_node.gradient, AbstractArray)
                parameter_node.gradient .= zero(el_type)
            else
                parameter_node.gradient = zero(el_type)
            end
        end
    end
end

# Collect all trainable parameters from the graph starting from a given node.
function parameters(start_node::Node)
    trainable_params_set = Set{Node}()
    visited_nodes = Set{Node}()
    # Stack for DFS
    nodes_to_visit_stack = [start_node]

    while !isempty(nodes_to_visit_stack)
        current_node = pop!(nodes_to_visit_stack)
        if current_node in visited_nodes
            continue
        end
        push!(visited_nodes, current_node)

        if current_node.is_trainable_param
            push!(trainable_params_set, current_node)
        end

        for input_node in current_node.inputs
            push!(nodes_to_visit_stack, input_node)
        end
    end
    
    return collect(trainable_params_set)
end