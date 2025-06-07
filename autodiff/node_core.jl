mutable struct Node{T<:Real}
    value::Union{T, AbstractArray{T}}
    gradient::Union{Nothing, T, AbstractArray{T}}
    inputs::Vector{Node}          
    backward_function::Function   
    is_trainable_param::Bool

    # Constructor for leaf nodes (input data, parameters)
    function Node(val::Union{T, AbstractArray{T}}; is_trainable::Bool=false) where {T<:Real}
        new{T}(val, nothing, Node[], () -> nothing, is_trainable)
    end

    # Constructor for internal nodes (results of operations)
    function Node(val::Union{T, AbstractArray{T}}, operation_inputs::Vector{Node}, bwd_func::Function) where {T<:Real}
        new{T}(val, nothing, operation_inputs, bwd_func, false)
    end
end

value(node::Node) = node.value
value(x::Real) = x

grad(node::Node) = node.gradient

_eltype(node::Node{T}) where T = T
_eltype(arr::AbstractArray{T}) where T = T
_eltype(x::T) where T<:Real = T

# Set or initializes the gradient for a Node.
function grad!(node::Node{T}, gradient_value) where T
    gradient_converted = if isa(gradient_value, AbstractArray)
        convert(AbstractArray{T}, gradient_value)
    elseif isa(gradient_value, Real)
        convert(T, gradient_value)
    else
        @error "Incorrect gradient type: $(typeof(gradient_value)). Expected Real or AbstractArray for Node $(node)."
        return
    end

    if node.gradient === nothing
        if isa(node.value, AbstractArray) && isa(gradient_converted, Real)
            node.gradient = fill(gradient_converted, size(value(node)))
        else
            node.gradient = deepcopy(gradient_converted)
        end
    else
        accumulate_gradient!(node, gradient_converted)
    end
end

Base.broadcastable(x::Node) = Ref(x)