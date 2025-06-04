function accumulate_gradient!(n::Node{T}, g_update) where {T<:Real}
    g_converted = if isa(g_update, AbstractArray)
        convert(AbstractArray{T}, g_update)

    elseif isa(g_update, Real)
        convert(T, g_update)

    else
        @error "It has to be Real or AbstractArray for Node $(n), not $(typeof(g_update))."
        return
    end

    if n.gradient === nothing
        n.gradient = deepcopy(g_converted)
    else
        if isa(n.gradient, Real) && isa(g_converted, Real)
            n.gradient += g_converted

        elseif isa(n.gradient, AbstractArray) && isa(g_converted, AbstractArray) && size(n.gradient) == size(g_converted)
            n.gradient .+= g_converted

        elseif isa(n.gradient, AbstractArray) && isa(g_converted, Real)
            n.gradient .+= g_converted

        elseif isa(n.gradient, Real) && isa(g_converted, AbstractArray)
            n.gradient += sum(g_converted) 

        elseif isa(n.gradient, AbstractArray) && isa(g_converted, AbstractArray)
            try
                n.gradient .+= g_converted
            catch e
                @error "Error: $(e) gradient accumulation for Node $(n). Size of existing gradient: $(size(n.gradient)), size of update: $(size(g_converted))."
            end

        else
            @error "Unsupported combination of types/shapes in accumulate_gradient! for Node $(n)."
        end
    end
end

function backward!(output_loss_node::Node{T}) where {T<:Real}
    if output_loss_node.gradient === nothing
        if isa(output_loss_node.value, Real) || length(output_loss_node.value) == 1
            grad!(output_loss_node, one(T))
        else
            error("""
            Błąd inicjalizacji propagacji wstecznej (backward!):
            Rozpoczęto na Node przechowującym wieloelementową tablicę $(size(output_loss_node.value)) typu $T
            bez ustawionego początkowego gradientu. Node: $(output_loss_node).
            Propagacja wsteczna powinna zaczynać się od skalarnego węzła straty.
            """)
        end
    elseif !(isa(output_loss_node.gradient, Real) || length(output_loss_node.gradient) == 1)
         @warn """
         Propagacja wsteczna (backward!) rozpoczęta na Node $(output_loss_node), którego gradient nie jest skalarem.
         Kształt gradientu: $(size(output_loss_node.gradient)). Upewnij się, że to zamierzone.
         """
    end

    processing_order = Node[]
    visited_for_topo = Set{Node}()

    function build_processing_order_dfs(curr_node::Node) # Zmieniono nazwę argumentu dla jasności
        push!(visited_for_topo, curr_node)
        for input_node_val in curr_node.inputs # Zmieniono nazwę pola
            if !(input_node_val in visited_for_topo)
                build_processing_order_dfs(input_node_val)
            end
        end
        push!(processing_order, curr_node)
    end

    build_processing_order_dfs(output_loss_node)
    
    for current_processing_node in reverse(processing_order) # Zmieniono nazwę zmiennej pętli
        if current_processing_node.gradient !== nothing
            current_processing_node.backward_function()
        else
            if current_processing_node !== output_loss_node
                 @debug """
                 Node $(current_processing_node) pominięty w backward pass (brak gradientu).
                 Może to być normalne, jeśli węzeł nie wpływa na ostateczną stratę.
                 """
            end
        end
    end
end

# --- Zero Gradients ---
"Zeruje gradienty dla podanej listy parametrów `Node`."
function zero_grad!(trainable_params::AbstractVector{<:Node})
    if isempty(trainable_params)
        @warn "Próba zerowania gradientów dla pustej listy parametrów. Nic nie zrobiono."
        return
    end
    for p_node in trainable_params # Zmieniono nazwę zmiennej pętli
        if !p_node.is_trainable_param
            @warn "Próba zerowania gradientu dla Node, który nie jest oznaczony jako parametr uczący się: $(p_node)."
        end
        
        if p_node.gradient !== nothing
            param_el_type = _eltype(p_node.value)
            if isa(p_node.gradient, AbstractArray)
                p_node.gradient .= zero(param_el_type)
            else
                p_node.gradient = zero(param_el_type)
            end
        end
    end
end

# --- Collect Parameters ---
"Zbiera wszystkie parametry (Node oznaczone jako is_trainable_param=true) z grafu, zaczynając od `v_start_node`."
function parameters(v_start_node::Node) # Zmieniono nazwę argumentu
    trainable_params_set = Set{Node}()
    visited_nodes = Set{Node}()
    nodes_to_visit_stack = [v_start_node]

    while !isempty(nodes_to_visit_stack)
        current_graph_node = pop!(nodes_to_visit_stack) # Zmieniono nazwę zmiennej

        if current_graph_node in visited_nodes
            continue
        end
        push!(visited_nodes, current_graph_node)

        if current_graph_node.is_trainable_param
            push!(trainable_params_set, current_graph_node)
        end

        for input_val_node in current_graph_node.inputs # Zmieniono nazwę zmiennej pętli i pola
            push!(nodes_to_visit_stack, input_val_node)
        end
    end
    return collect(trainable_params_set)
end