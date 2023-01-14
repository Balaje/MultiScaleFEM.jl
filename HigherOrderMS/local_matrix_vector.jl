#########################################################
# Functions to generate the local matrix-vector systems
# Does not depend on the order of polynomial
#########################################################

"""
Function to get the local matrix-vector system corresponding to the inner products:
    (u,v) = ∫ₖ u*v dx
    a(u,v) = ∫ₖ ∇(u)⋅∇(v) dx
    (f,v)  = ∫ₖ f*v dx
Here u,v ∈ H¹₀(K) and f is a known function.
"""
function _local_matrix_H¹_H¹(xn, ϕ̂::Function, A::Function, quad, h, p)
  Me = Array{Float64}(undef, p+1, p+1)
  Ke = Array{Float64}(undef, p+1, p+1)
  fill!(Me, 0.)
  fill!(Ke, 0.)
  qs,ws = quad
  J = 0.5*h
  for q=1:lastindex(qs)
    x̂ = qs[q]
    x = (xn[2] + xn[1])*0.5 .+ 0.5*h*x̂
    res = ϕ̂(x̂)
    res1 = ∇(ϕ̂, x̂)
    # Loop over the local matrices
    for i=1:p+1, j=1:p+1
        Me[i,j] += ws[q]*( res[i] * res[j] )*J
        Ke[i,j] += ws[q]*( A(x) * res1[i] * res1[j] )*J^-1
    end
  end
  Me, Ke
end
function _local_vector_H¹(xn, ϕ̂::Function, f::Function, quad, h, p)
  Fe = Vector{Float64}(undef, p+1)
  fill!(Fe, 0.)
  qs,ws = quad
  J = 0.5*h
  for q=1:lastindex(qs)
    x̂ = qs[q]
    x = (xn[2]+xn[1])*0.5 .+ 0.5*h*x̂
    res = ϕ̂(x̂)
    for i=1:p+1
      Fe[i] += ws[q]*( f(x)*res[i] )*J
    end
  end
  Fe
end
"""
Function to get the rectangular matrix corresponding to the inner product:
    (u,Λₖ) = ∫ₕ u*Λₖ dx
Here u ∈ H¹₀(K), v ∈ Vₕᵖ(K)
"""
function _local_matrix_H¹_Vₕᵖ(nds, A::Function, basis::Tuple{Function,Function},
                              quad, hlocal, fespace)
  # nds should be the fine-scale interval
  qs, ws = quad
  q,p = fespace
  Me = Matrix{Float64}(undef, q+1, p+1)
  fill!(Me,0.0)
  ϕ̂,Λₖ = basis
  for qⱼ=1:lastindex(qs)
    x̂ = (nds[1]+nds[2])/2 + (nds[2]-nds[1])/2*qs[qⱼ]
    for i=1:q+1, j=1:p+1
      Me[i,j] += ws[qⱼ]*A(x̂)*Λₖ(x̂)[j]*ϕ̂(x̂)[i]
    end
  end
  Me*hlocal*0.5
end
function _local_vector_Vₕᵖ(nds, f::Function, Λₖ::Function, quad, hlocal, p)
  qs, ws = quad
  Fe = Vector{Float64}(undef,p+1)
  fill!(Fe,0.0)
  for qⱼ=1:lastindex(qs)
    x̂ = (nds[1]+nds[2])/2 + (nds[2]-nds[1])/2*qs[qⱼ]
    for i=1:p+1
      Fe[i] += ws[qⱼ]*f(x̂)*Λₖ(x̂)[i]
    end
  end
  Fe*0.5*hlocal
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
