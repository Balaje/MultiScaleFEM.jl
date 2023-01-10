##########################################################################################################
# Functions to generate the local matrix-vector systems (Uses the Gridap solution for the local basis)
##########################################################################################################

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
      Fe[i] += ws[qp] * f(x) * Λ̃ₖˡ(x, R[i]) * J
      for j=1:dim
        Ke[i,j] += ws[qp] * A(x) * ∇Λ̃ₖˡ(x, R[i])* ∇Λ̃ₖˡ(x, R[j]) * J
        Me[i,j] += ws[qp] * Λ̃ₖˡ(x, R[i]) * Λ̃ₖˡ(x, R[j]) * J
      end
    end
  end  
  Ke, Me, Fe
end