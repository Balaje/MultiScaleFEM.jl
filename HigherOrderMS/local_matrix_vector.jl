#########################################################
# Functions to generate the local matrix-vector systems
# Does not depend on the order of polynomial
#########################################################

"""
Function to get the local matrix-vector system corresponding to the inner products:
    (u,v) = ∫ₖ u*v dx
    (f,v)  = ∫ₖ f*v dx
Here u,v ∈ H¹₀(K) and f is a known function.
"""
function _local_matrix!(Me, xn, basis::Tuple{Function,Function}, A::Function, quad, h, fespace)    
  fill!(Me, 0.)
  qs,ws = quad
  q,p = fespace
  basis_1, basis_2 = basis
  for qk=1:lastindex(qs)
    x̂ = qs[qk]
    x = (xn[2]+xn[1])*0.5 .+ 0.5*h*x̂    
    # Loop over the local matrices
    for i=1:q+1, j=1:p+1
        Me[i,j] += ws[qk]*( A(x̂) * basis_1(x̂)[i] * basis_2(x̂)[j] )      
    end
  end
  Me*0.5*h
end
function _local_vector!(Fe, xn, basis::Function, f::Function, quad, h, fespace)
  fill!(Fe, 0.)
  qs,ws = quad
  p = fespace
  for q=1:lastindex(qs)
    x̂ = qs[q]
    x = (xn[2]+xn[1])*0.5 .+ 0.5*h*x̂
    for i=1:p+1
      Fe[i] += ws[q]*( f(x) * basis(x̂)[i] )*J
    end
  end
  Fe
end



########################################################################################################################################
"""
Function to compute the element matrix-vector system for the multiscale method:
  a(uᵐˢ, vᵐˢ) = ∫ₖ A(x) ∇(uᵐˢ)⋅∇(vᵐˢ) dx
  (uᵐˢ, vᵐˢ) = ∫ₖ (uᵐˢ)*(vᵐˢ) dx
  (f, vᵐˢ) = ∫ₖ (f)*(vᵐˢ)dx
where uᵐˢ,vᵐˢ ∈ Ṽₕᵖˡ and f is a known function.
"""
  function _local_matrix_vector_MS_MS(xn, A::Function, f::Function, quad, H, fespace, dim, R::VecOrMat{Rˡₕ})
    Ke = Array{Float64}(undef, dim, dim)
    Fe = Vector{Float64}(undef, dim)
    Me = Array{Float64}(undef, dim, dim)
    @assert dim == length(R)
    q,p=fespace
    fill!(Ke, 0.); fill!(Fe, 0.); fill!(Me,0.)
    qs, ws = quad
    J = 0.5*H
    for qp=1:lastindex(qs)
      x̂ = qs[qp]
      x = (xn[2]+xn[1])*0.5 + 0.5*H*x̂
      for i=1:dim
        Fe[i] += ws[qp] * f(x) * Λ̃ₖˡ(x, R[i], fespace) * J
        for j=1:dim
          Ke[i,j] += ws[qp] * A(x) * ∇Λ̃ₖˡ(x, R[i], fespace)* ∇Λ̃ₖˡ(x, R[j], fespace) * J
          Me[i,j] += ws[qp] * Λ̃ₖˡ(x, R[i], fespace) * Λ̃ₖˡ(x, R[j], fespace) * J
        end
      end
    end
    Ke, Me, Fe
  end
  ########################################################################################################################################
