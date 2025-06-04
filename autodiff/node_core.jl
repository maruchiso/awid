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

value(n::Node) = n.value
value(x::Real) = x

grad(n::Node) = n.gradient

_eltype(n::Node{T}) where T = T
_eltype(arr::AbstractArray{T}) where T = T
_eltype(x::T) where T<:Real = T

# Set or initializes the gradient for a Node.
function grad!(n::Node{T}, g_val) where T
    g_converted = if isa(g_val, AbstractArray)
        convert(AbstractArray{T}, g_val)
    elseif isa(g_val, Real)
        convert(T, g_val)
    else
        @error "Incorrect gradient type: $(typeof(g_val)). Expected Real or AbstractArray for Node $(n)."
        return
    end

    if n.gradient === nothing
        if isa(n.value, AbstractArray) && isa(g_converted, Real)
            n.gradient = fill(g_converted, size(value(n)))
        else
            n.gradient = deepcopy(g_converted)
        end
    else
        accumulate_gradient!(n, g_converted)
    end
end

Base.broadcastable(x::Node) = Ref(x)