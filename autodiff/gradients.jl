function accumulate_gradient!(n::Node{T}, gradient_update) where {T<:Real}
    gradient_converted = if isa(gradient_update, AbstractArray)
        convert(AbstractArray{T}, gradient_update)

    elseif isa(gradient_update, Real)
        convert(T, gradient_update)

    else
        @error "It has to be Real or AbstractArray for Node $(n), not $(typeof(gradient_update))."
        return
    end

    if n.gradient === nothing
        n.gradient = deepcopy(gradient_converted)
    else
        if isa(n.gradient, Real) && isa(gradient_converted, Real)
            n.gradient += gradient_converted

        elseif isa(n.gradient, AbstractArray) && isa(gradient_converted, AbstractArray) && size(n.gradient) == size(gradient_converted)
            n.gradient .+= gradient_converted

        elseif isa(n.gradient, AbstractArray) && isa(gradient_converted, Real)
            n.gradient .+= gradient_converted

        elseif isa(n.gradient, Real) && isa(gradient_converted, AbstractArray)
            n.gradient += sum(gradient_converted) 

        elseif isa(n.gradient, AbstractArray) && isa(gradient_converted, AbstractArray)
            try
                n.gradient .+= gradient_converted
            catch e
                @error "Error: $(e) gradient accumulation for Node $(n). Size of existing gradient: $(size(n.gradient)), size of update: $(size(gradient_converted))."
            end

        else
            @error "Unsupported combination of types/shapes in accumulate_gradient! for Node $(n)."
        end
    end
end

function backward!(output_loss_node::Node{T}) where {T<:Real}
    # Initialize gradient
    if output_loss_node.gradient === nothing
        if isa(output_loss_node.value, Real) || length(output_loss_node.value) == 1 
            grad!(output_loss_node, one(T))
        else
            error("Backward pass started on Node $(output_loss_node), which has no gradient set.")
        end

    elseif !(isa(output_loss_node.gradient, Real) || length(output_loss_node.gradient) == 1)
         @warn "Backward pass started on Node $(output_loss_node), which has a gradient that is not a scalar. Gradient shape: $(size(output_loss_node.gradient))."
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
        push!(processing_order, current_node)
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
        @warn "Tryed to zero gradients for an empty list of trainable parameters."
        return
    end
    for parameter_node in trainable_params 
        if !parameter_node.is_trainable_param
            @warn "Zero grad for Node $(parameter_node) that is not a trainable parameter."
        end
        
        if parameter_node.gradient !== nothing
            param_el_type = _eltype(parameter_node.value)
            if isa(parameter_node.gradient, AbstractArray)
                parameter_node.gradient .= zero(param_el_type)
            else
                parameter_node.gradient = zero(param_el_type)
            end
        end
    end
end


# Collect all trainable parameters from the graph starting from a given node.
function parameters(start_node::Node) 
    trainable_params_set = Set{Node}()
    visited_nodes = Set{Node}()
    nodes_to_visit_stack = [start_node] # Stack for DFS

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