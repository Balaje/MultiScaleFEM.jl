###################################################
# Essential functions for the multiscale method
# DO NOT MODIFY
###################################################

"""
Function to generate the 1d basis functions (Multiscale version)
"""
function ms_basis_den(A, xn; order=40)
    qs,ws = gausslegendre(order)
    den = 0.
    J_den = abs(xn[2]-xn[1])/2
    for q = 1:order
        xq = qs[q]
        x_den = (xn[2]+xn[1])/2 .+ (xn[2]-xn[1])/2*xq
        den += ws[q]*(A(x_den))^(-1)*J_den
    end 
    den
end 
function φ̂ₘₛ(x, A::Function, xn; order=40)
    qs,ws = gausslegendre(order)
    res = [0.,0.]
    num = 0.    
    #x̂ = (xn[1] + xn[2])/2 .+ (xn[2] - xn[1])/2*x[1] # Transform to local coordinates
    for q = 1:order
        xq = qs[q]        
        # Numerator integral
        J_num = abs(x - xn[1])/2  # Jacobian of the numerator
        x_num = (x + xn[1])/2 .+ (x - xn[1])/2*xq                                        
        num += ws[q]*(A(x_num))^(-1)*J_num
    end 
    den = ms_basis_den(A, xn; order=order)  # Get the denominator integral   
    # Basis function 1 and 2    
    res[1] = num/den
    res[2] = 1 - num/den    
    res
end 
function ∇φ̂ₘₛ(x, A::Function, xn; order=40)
    res = [0., 0.]
    den = ms_basis_den(A, xn; order=order)
    num = (A(x))^(-1)
    res[1] = num/den
    res[2] = -num/den
    res
end
function φ̂ₘₛ!(res, x, A::Function, xn; order=40)
    qs,ws = gausslegendre(order)
    num = 0.   
    for q = 1:order
        xq = qs[q]
        x_num = (x + xn[1])/2 .+ (x - xn[1])/2*xq                
        J_num = abs(x - xn[1])/2        
        num += ws[q]*(A(x_num))^(-1)*J_num
    end 
    den = ms_basis_den(A, xn; order=order)  # Get the denominator integral
    # Basis function 1 and 2    
    res[1] = num/den
    res[2] = 1 - num/den    
end
function ∇φ̂ₘₛ!(res, x, A::Function, xn; order=40)
    den = ms_basis_den(A, xn; order=order)
    num = (A(x))^(-1)
    res[1] = num/den
    res[2] = -num/den
end


""" 
Function to compute the local stiffness and mass matrix (multiscale version)
"""
function local_matrix_vector_multiscale!(cache, xn, A::Function, f::Function, quad)
    Me, Ke, Fe, res, res1 = cache
    fill!(Me, 0.); fill!(Ke, 0.); fill!(Fe, 0.);
    fill!(res, 0.); fill!(res1, 0.)
  
    qs,ws = quad
    J = (xn[2] - xn[1])/2
  
    for q=1:lastindex(qs)
      x̂ = qs[q]
      x = (xn[2] + xn[1])/2 .+ (xn[2] - xn[1])/2*x̂
      φ̂ₘₛ!(res, x, A, xn; order=order)
      ∇φ̂ₘₛ!(res1, x, A, xn; order=order)
      for i=1:2
        ϕᵢᵐˢ = res[i]
        ∇ϕᵢᵐˢ = res1[i]
        Fe[i] += ws[q]*( f(x)*ϕᵢᵐˢ )*J
        for j=1:2
          ϕⱼᵐˢ = res[j]
          ∇ϕⱼᵐˢ = res1[j]
          Me[i,j] += ws[q]*( ϕᵢᵐˢ * ϕⱼᵐˢ )*J
          Ke[i,j] += ws[q]*( A(x) * ∇ϕᵢᵐˢ * ∇ϕⱼᵐˢ )*J
        end
      end
    end
end

"""
Function to compute the L² error in the solution.
(Cannot use the old method since ũₕ(xⱼ) = Iₕu(xⱼ) = u(xⱼ) here)
(So evaluate the finite element functions along with the exact function on the quadrature points)
"""
function l2err_multiscale(uh::AbstractVector, u::Function, nodes, elem, A::Function; quad=gausslegendre(6))
    qs,ws = quad
    J = (nodes[2]-nodes[1])*0.5
    nel = size(elem,1)
    nverts = size(nodes,1)
    function uhx(x)
        val = 0
        for t=2:nverts-1
            if(nodes[t-1] ≤ x ≤  nodes[t])                              
                val = val + uh[t]*φ̂ₘₛ(x[1], A, [nodes[t-1], nodes[t]]; order=length(qs))[1]
            elseif(nodes[t] ≤ x ≤ nodes[t+1])                
                val = val + uh[t]*φ̂ₘₛ(x[1], A, [nodes[t], nodes[t+1]]; order=length(qs))[2]
            else
                val = val + 0
            end 
        end 
        val
    end 
    err = 0
    res = zeros(Float64,2)
    for t=1:nel
        nds = nodes[elem[t,:]]
        for q=1:lastindex(qs)
            x̂ = qs[q]
            uhx̂ = uhx((nds[2]+nds[1])/2 + (nds[2]-nds[1])/2*x̂)            
            ux̂ = u((nds[2]+nds[1])/2 + (nds[2]-nds[1])/2*x̂)            
            err += ws[q]*(ux̂ - uhx̂)^2*J
        end
    end

    sqrt(err)
end