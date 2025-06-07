function sum_to(x_grad::AbstractArray{T}, target_size::Tuple) where T
    if size(x_grad) == target_size
        return x_grad
    end

    if isempty(target_size) || all(d == 1 for d in target_size)
        return sum(x_grad)::T
    end

    ndims_x = ndims(x_grad)
    ndims_target = length(target_size)
    dims_to_sum = Int[]

    for d = 1:ndims_x
        if d > ndims_target || (target_size[d] == 1 && size(x_grad, d) > 1)
            push!(dims_to_sum, d)
        elseif d <= ndims_target && target_size[d] != 1 && size(x_grad,d) != 1 && size(x_grad, d) != target_size[d]
             error("Dimension $d: $(size(x_grad, d)) != $(target_size[d])")
        end
    end
    
    result = isempty(dims_to_sum) ? x_grad : sum(x_grad, dims=tuple(dims_to_sum...))

    return if size(result) == target_size
        result
    else
        try
            reshape(result, target_size)
        catch e
            error("Cannot reshape gradient from $(size(result)) to $(target_size).")
        end
    end
end

function sum_to(x_scalar_grad::T, target_size::Tuple) where T<:Real
    return fill(x_scalar_grad, target_size)::AbstractArray{T}
end
function sum_to(x_scalar_grad::T, target_size::Tuple{}) where T<:Real
    return x_scalar_grad::T
end


function Base.:+(a::Node{T}, b::Node{T}) where {T<:Real}
    val_a = value(a)
    val_b = value(b)
    result_val = val_a .+ val_b 
    op_inputs = Node[a, b]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (+) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        grad_a_unshaped = output_grad
        grad_b_unshaped = output_grad
        
        grad_a_final = grad_a_unshaped
        grad_b_final = grad_b_unshaped
        if size(val_a) != size(output_grad); grad_a_final = sum_to(output_grad, size(val_a)); end
        if size(val_b) != size(output_grad); grad_b_final = sum_to(output_grad, size(val_b)); end

        accumulate_gradient!(a, grad_a_final)
        accumulate_gradient!(b, grad_b_final)
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end
Base.:+(a::Node{T}, b_val::Real) where T = a + Node(fill(T(b_val), size(value(a))); is_trainable=false)
Base.:+(a_val::Real, b::Node{T}) where T = Node(fill(T(a_val), size(value(b))); is_trainable=false) + b


function Base.:-(a::Node{T}, b::Node{T}) where {T<:Real}
    val_a = value(a)
    val_b = value(b)
    result_val = val_a .- val_b 
    op_inputs = Node[a, b]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (-) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        grad_a_val = output_grad
        grad_b_val = -output_grad
        
        grad_a_final = grad_a_val
        grad_b_final = grad_b_val
        if size(val_a) != size(output_grad); grad_a_final = sum_to(grad_a_val, size(val_a)); end
        if size(val_b) != size(output_grad); grad_b_final = sum_to(grad_b_val, size(val_b)); end
        
        accumulate_gradient!(a, grad_a_final)
        accumulate_gradient!(b, grad_b_final)
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end
Base.:-(a::Node{T}, b_val::Real) where T = a - Node(fill(T(b_val), size(value(a))); is_trainable=false)
Base.:-(a_val::Real, b::Node{T}) where T = Node(fill(T(a_val), size(value(b))); is_trainable=false) - b

function Base.:-(a::Node{T}) where {T<:Real}
    val_a = value(a)
    result_val = -val_a
    op_inputs = Node[a]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (unary -) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        grad_a_val = -output_grad
        
        grad_a_final = grad_a_val
        if size(val_a) != size(output_grad); grad_a_final = sum_to(grad_a_val, size(val_a)); end
        accumulate_gradient!(a, grad_a_final)
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end

function Base.:*(a::Node{T}, b::Node{T}) where {T<:Real}
    val_a = value(a)
    val_b = value(b)
    result_val = val_a .* val_b
    op_inputs = Node[a, b]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (*) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        grad_a_unshaped = output_grad .* val_b
        grad_b_unshaped = output_grad .* val_a
        
        accumulate_gradient!(a, sum_to(grad_a_unshaped, size(val_a)))
        accumulate_gradient!(b, sum_to(grad_b_unshaped, size(val_b)))
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end
Base.:*(a::Node{T}, b_val::Real) where T = a * Node(fill(T(b_val), size(value(a))); is_trainable=false)
Base.:*(a_val::Real, b::Node{T}) where T = Node(fill(T(a_val), size(value(b))); is_trainable=false) * b

function matmul(a::Node{T}, b::Node{T}) where T
    val_a = value(a)
    val_b = value(b)
    
    dim_a_inner = isa(val_a, AbstractVector) ? length(val_a) : size(val_a, 2)
    dim_b_inner = isa(val_b, AbstractVector) ? length(val_b) : size(val_b, 1)

    if dim_a_inner != dim_b_inner
        error("Niekompatybilne wymiary wewnętrzne dla matmul Node $(a) i Node $(b): $(size(val_a)) (wewn. $(dim_a_inner)) i $(size(val_b)) (wewn. $(dim_b_inner))")
    end

    result_val = val_a * val_b
    op_inputs = Node[a, b]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (matmul) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        grad_a = output_grad * transpose(val_b)
        grad_b = transpose(val_a) * output_grad
        
        accumulate_gradient!(a, grad_a)
        accumulate_gradient!(b, grad_b)
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end

function Base.:/(a::Node{T}, b::Node{T}) where {T<:Real}
    val_a = value(a)
    val_b = value(b)
    eps_T = T(1e-8)
    denom_stable = val_b .+ eps_T
    result_val = val_a ./ denom_stable
    op_inputs = Node[a, b]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (/) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        grad_a_unshaped = output_grad ./ denom_stable
        grad_b_unshaped = -output_grad .* val_a ./ (denom_stable .^ 2)
        
        accumulate_gradient!(a, sum_to(grad_a_unshaped, size(val_a)))
        accumulate_gradient!(b, sum_to(grad_b_unshaped, size(val_b)))
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end
Base.:/(a::Node{T}, b_val::Real) where T = a / Node(fill(T(b_val), size(value(a))); is_trainable=false)
Base.:/(a_val::Real, b::Node{T}) where T = Node(fill(T(a_val), size(value(b))); is_trainable=false) / b

function Base.:^(a::Node{T}, n_val::Real) where {T<:Real}
    val_a = value(a)
    n_T = T(n_val)
    if any(x -> x < 0 && (n_T != round(n_T) && n_T < 1.0), val_a)
         @warn "Potęgowanie ujemnej podstawy w Node $(a) do wykładnika $(n_T) może prowadzić do problemów."
    end
    result_val = val_a .^ n_T
    op_inputs = Node[a]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (^) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        
        eps_T_pow = Base.eps(T)
        _sign_val_a = isa(val_a, AbstractArray) ? sign.(val_a) : sign(val_a)
        base_stable_for_grad = val_a .+ (T.(_sign_val_a) .* eps_T_pow)

        grad_a_unshaped = if n_T == zero(T)
            zeros_like_val_a = similar(val_a, T); fill!(zeros_like_val_a, zero(T))
            output_grad .* zeros_like_val_a
        else
            term_pow = n_T - one(T)
            pow_component = ifelse.( (val_a .== zero(T)) .& (n_T .> one(T)), 
                                     zero(T), 
                                     base_stable_for_grad .^ term_pow)
            output_grad .* n_T .* pow_component
        end
        
        accumulate_gradient!(a, sum_to(grad_a_unshaped, size(val_a)))
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end

function Base.exp(a::Node{T}) where {T<:Real}
    val_a = value(a)
    result_val = exp.(val_a)
    op_inputs = Node[a]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (exp) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        grad_a_unshaped = output_grad .* result_val
        accumulate_gradient!(a, sum_to(grad_a_unshaped, size(val_a)))
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end

function Base.log(a::Node{T}; ϵ::Union{Nothing,Real}=nothing) where {T<:Real}
    val_a = value(a)
    effective_eps = T(ϵ === nothing ? Base.eps(T(1.0)) : ϵ)
    if any(x -> x <= zero(T), val_a)
        @warn "Logarytmowanie niedodatniej wartości w Node $(a). Użyto epsilon $(effective_eps) dla stabilizacji."
    end
    value_stable_for_log = max.(val_a, effective_eps)
    result_val = log.(value_stable_for_log)
    op_inputs = Node[a]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (log) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        grad_a_unshaped = output_grad ./ value_stable_for_log
        accumulate_gradient!(a, sum_to(grad_a_unshaped, size(val_a)))
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end

function Base.max(a::Node{T}, scalar_val::Real) where {T<:Real}
    val_a = value(a)
    val_T_scalar = T(scalar_val)
    result_val = max.(val_a, val_T_scalar)
    op_inputs = Node[a]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (max) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        mask = T.(val_a .> val_T_scalar)
        grad_a_unshaped = output_grad .* mask
        accumulate_gradient!(a, sum_to(grad_a_unshaped, size(val_a)))
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end
Base.max(scalar_val::Real, a::Node{T}) where T = Base.max(a, scalar_val)

function Base.sum(a::Node{T}) where {T<:Real}
    val_a = value(a)
    result_val = sum(val_a)::T
    op_inputs = Node[a]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (sum) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        if !isa(output_grad, Real)
             @warn "Oczekiwano skalarnego gradientu dla operacji sum dla Node $(new_output_node), otrzymano $(typeof(output_grad)) o rozmiarze $(size(output_grad))."
        end
        grad_a_filled = fill(output_grad, size(val_a))
        accumulate_gradient!(a, grad_a_filled)
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end

function relu(a::Node{T}) where {T<:Real}
    val_a = value(a)
    result_val = max.(val_a, zero(T))
    op_inputs = Node[a]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (relu) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        mask = T.(val_a .> zero(T))
        grad_a_unshaped = output_grad .* mask
        accumulate_gradient!(a, sum_to(grad_a_unshaped, size(val_a)))
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end

function sigmoid(a::Node{T}) where {T<:Real}
    val_a = value(a)
    result_val = similar(val_a, T)
    for i in eachindex(val_a)
        x_val_i = val_a[i]
        if x_val_i >= zero(T)
            result_val[i] = one(T) / (one(T) + exp(-x_val_i))
        else
            exp_x_val_i = exp(x_val_i)
            result_val[i] = exp_x_val_i / (exp_x_val_i + one(T))
        end
    end
    op_inputs = Node[a]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (sigmoid) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        s_val = result_val
        grad_a_unshaped = output_grad .* s_val .* (one(T) .- s_val)
        accumulate_gradient!(a, sum_to(grad_a_unshaped, size(val_a)))
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end

function Base.tanh(a::Node{T}) where {T<:Real}
    val_a = value(a)
    result_val = tanh.(val_a)
    op_inputs = Node[a]
    local new_output_node

    function backward_function_impl()
        output_grad = grad(new_output_node)
        if output_grad === nothing
            @error "Krytyczny błąd w backward (tanh) : gradient wyjścia jest 'nothing' dla Node $(new_output_node)."
            return
        end
        grad_a_unshaped = output_grad .* (one(T) .- result_val .^ 2)
        accumulate_gradient!(a, sum_to(grad_a_unshaped, size(val_a)))
    end
    new_output_node = Node(result_val, op_inputs, backward_function_impl)
    return new_output_node
end