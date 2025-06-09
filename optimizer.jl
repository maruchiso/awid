module Optimizers

using ..AutoDiff

export SGD, Adam, update!

mutable struct SGD{T<:Real}
    learning_rate::T 
    params::Vector{<:Node{T}}

    function SGD(model_params::Vector{<:Node{T}}, learning_rate::T) where {T<:Real}
        for p in model_params
            if !p.is_trainable_param
                @warn "Optimizer SGD received a parameter that is not marked as trainable: $(p)"
            end
        end
        new{T}(learning_rate, model_params)
    end
end

function update!(opt::SGD{T}) where {T<:Real}
    for p_node in opt.params
        if p_node.is_trainable_param && p_node.gradient !== nothing
            gradient_val = grad(p_node)
            
            if isa(p_node.value, AbstractArray)
                p_node.value .-= opt.learning_rate .* gradient_val
            else
                p_node.value -= opt.learning_rate * gradient_val
            end
        end
    end
end

mutable struct Adam{T<:Real}
    lr::T
    beta1::T
    beta2::T
    epsilon::T
    params::Vector{<:Node{T}}
    m::Dict{Node, Any} # Dict for 1st moment m
    v::Dict{Node, Any} # Dict for 2nd moment v
    t::Int  # steps counter

    function Adam(params::Vector{<:Node{T}}; lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8) where T
        # Conver learning rate and beta parameters to the specified type T
        lr_T, b1_T, b2_T, eps_T = T(lr), T(beta1), T(beta2), T(epsilon)
        
        m = Dict{Node, Any}()
        v = Dict{Node, Any}()
        for p in params
            if !p.is_trainable_param
                @warn "Optimizer Adam received a parameter that is not marked as trainable: $(p)"
            end
            m[p] = zeros(T, size(value(p)))
            v[p] = zeros(T, size(value(p)))
        end
        
        new{T}(lr_T, b1_T, b2_T, eps_T, params, m, v, 0)
    end
end

function update!(opt::Adam{T}) where T
    opt.t += 1
    
    for p in opt.params
        if p.is_trainable_param && p.gradient !== nothing
            g = grad(p)
            
            opt.m[p] = opt.beta1 .* opt.m[p] .+ (1 - opt.beta1) .* g
            
            opt.v[p] = opt.beta2 .* opt.v[p] .+ (1 - opt.beta2) .* (g .^ 2)
            
            m_hat = opt.m[p] ./ (1 - opt.beta1^opt.t)
            v_hat = opt.v[p] ./ (1 - opt.beta2^opt.t)
            
            update_val = opt.lr .* m_hat ./ (sqrt.(v_hat) .+ opt.epsilon)
            
            if isa(p.value, AbstractArray)
                p.value .-= update_val
            else
                p.value -= update_val
            end
        end
    end
end

end