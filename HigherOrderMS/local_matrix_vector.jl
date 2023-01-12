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
function _local_matrix_vector_H¹_H¹(xn, A::Function, f::Function, quad, h, p)
  Me = Array{Float64}(undef, p+1, p+1)
  Ke = Array{Float64}(undef, p+1, p+1)
  Fe = Vector{Float64}(undef, p+1)
  res = Vector{Float64}(undef, p+1)
  res1 = Vector{Float64}(undef, p+1)
  fill!(Me, 0.); fill!(Ke, 0.); fill!(Fe, 0.);
  fill!(res,0.); fill!(res1,0)
  qs,ws = quad
  J = 0.5*h
  for q=1:lastindex(qs)
    x̂ = qs[q]
    x = (xn[2] + xn[1])*0.5 .+ 0.5*h*x̂
    res = ϕ̂(x̂,p)
    res1 = ∇ϕ̂(x̂,p)
    # Loop over the local matrices
    for i=1:p+1
      ϕᵢ = res[i]
      ∇ϕᵢ = res1[i]
      Fe[i] += ws[q]*( f(x)*ϕᵢ )*J
      for j=1:p+1
        ϕⱼ = res[j]
        ∇ϕⱼ = res1[j]
        Me[i,j] += ws[q]*( ϕᵢ * ϕⱼ )*J
        Ke[i,j] += ws[q]*( A(x) * ∇ϕᵢ * ∇ϕⱼ )*J^-1
      end
    end
  end
  Me, Ke, Fe
end
"""
Function to get the rectangular matrix corresponding to the inner product:
    (u,Λₖ) = ∫ₕ u*Λₖ dx
Here u ∈ H¹₀(K), v ∈ Vₕᵖ(K)
"""
function _local_matrix_H¹_Vₕᵖ(Λₖ::Function, nds::Tuple, fespace; h=2, qorder=10)
  # nds should be the fine-scale interval
  qs, ws = gausslegendre(qorder)
  q,p = fespace
  res = Matrix{Float64}(undef, q+1, p+1)
  fill!(res,0.0)
  for qⱼ=1:lastindex(qs)
    x̂ = (nds[1]+nds[2])/2 + (nds[2]-nds[1])/2*qs[qⱼ]
    for i=1:q+1, j=1:p+1
      res[i,j] += ws[qⱼ]*Λₖ(x̂)[j]*ϕ̂(x̂,q)[i]
    end
  end
  res*h*0.5
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
    display(R)
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
        Fe[i] += ws[qp] * f(x) * Λ̃ₖˡ(x, R[i]; fespace=fespace) * J
        for j=1:dim
          Ke[i,j] += ws[qp] * A(x) * ∇Λ̃ₖˡ(x, R[i]; fespace=fespace)* ∇Λ̃ₖˡ(x, R[j]; fespace=fespace) * J
          Me[i,j] += ws[qp] * Λ̃ₖˡ(x, R[i]) * Λ̃ₖˡ(x, R[j]) * J
        end
      end
    end
    Ke, Me, Fe
  end
  ########################################################################################################################################
