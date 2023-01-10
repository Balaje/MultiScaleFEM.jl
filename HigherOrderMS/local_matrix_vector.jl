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
    (u,v) = ∫ₖ u*v dx
Here u ∈ H¹₀(K), v ∈ L²(K)
"""    
function _localmassmatrix_H¹_L²((p₁,p₂); h=2)
  qopt = p₂+2
  qs,ws = gausslegendre(qopt)
  m,n=p₁,p₂
  res = Matrix{Float64}(undef,m+1,n+1)
  fill!(res, 0.)
  for i=1:m+1, j=1:n+1, q=1:qopt
    res[i,j] += ws[q] * ϕ̂(qs[q],p₁)[i] * ψ̂(qs[q],p₂)[j]
  end
  res = res*(0.5*h)
end
"""
Function to get the local matrix corresponding to the inner product:
    (u,v) = ∫ₖ u*v dx
Here u,v ∈ L²(K)
"""    
function _localmassmatrix_L²_L²(p::Int64; h=2)
  qopt = p+2
  qs,ws = gausslegendre(qopt)
  m,n=p+1,p+1
  res = Matrix{Float64}(undef,m,n)
  fill!(res, 0.)
  for i=1:m, j=1:n, q=1:qopt
    res[i,j] += ws[q] * ψ̂(qs[q],p)[i] * ψ̂(qs[q],p)[j]
  end
  res = res*(0.5*h)
end

"""
Function to get the local matrix corresponding to the inner product:
    (f,v) = ∫ₖ f*v dx
Here v ∈ L²(K) and f is a known function.
"""  
function _localvector_L²(xn, p::Int64, f::Function; qorder=10, h=2)
  quad = gausslegendre(qorder)
  qs, ws = quad
  qorder = length(qs)
  res = Vector{Float64}(undef,p+1)
  fill!(res,0.0)
  J = h*0.5
  for j=1:p+1
    for q=1:qorder
      x̂ = (xn[2]+xn[1])*0.5 .+ (h)*0.5*qs[q]
      res[j] += ws[q] * f(x̂) * ψ̂(qs[q],p)[j] * J
    end
  end
  res
end

########################################################################################################################################
########################################################################################################################################
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
########################################################################################################################################
########################################################################################################################################