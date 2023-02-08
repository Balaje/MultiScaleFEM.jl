"""
Basis function for the discontinuous space. Returns the legendre polynomials upto order p
  evaluated at x::Float64
  > res = Vector{Float64}(undef,p+1)
  > Λₖ!(res, x, element::Vector{Float64}, p::Int64)
  > res # contains the result
"""
function Λₖ!(res::AbstractVector{Float64}, x::Float64, nds::AbstractVector{Float64}, p::Int64)
  a,b = nds  
  fill!(res,0.0)
  if(((x-a) ≥ 1e-10) && ((x-b) ≤ 1e-10))
    x̂ = -(b+a)/(b-a) + 2.0*x/(b-a)  
    if(p==0)
      res[1] = 1.0
    elseif(p==1)
      res[1] = 1.0
      res[2] = x̂
    else      
      res[1] = 1.0
      res[2] = x̂
      for j=2:p
        res[j+1] = (2j-1)/(j)*x̂*res[j] - (j-1)/(j)*res[j-1]  
      end
    end
  else
    return
  end
end 
# Basis function of Vₕᵖ(K)
function fₖ!(res::Vector{Float64}, x::Float64, j::Int64, 
  nds::AbstractVector{Float64}, elem_coarse::Matrix{Int64}, el::Int64)      
  nodes = view(nds, view(elem_coarse,el,:))
  Λₖ!(res, x, nodes, p)
  res[j]
end 
"""
Standard Lagrange basis function of order p evaluated at x
  > cache = basis_cache(p)
  > ϕᵢ!(cache, x)
  > cache[3] # Contains the basis functions
"""
function basis_cache(p::Int64)
  xq = LinRange(-1,1,p+1)  
  Q = [xq[i]^j for i=1:p+1, j=0:p]
  A = Q\(I(p+1))
  b = Vector{Float64}(undef,p+1)
  fill!(b, 0.0)
  res = similar(b)
  fill!(res,0.0)
  return A', b, res
end
function ϕᵢ!(cache, x)
  A, b, res = cache
  fill!(res,0.0)
  q = length(res)
  for i=0:q-1
    b[i+1] = x^i
  end 
  mul!(res, A, b)
end
"""
First derivative of the Lagrange basis function of order p evaluated at x
  > cache = basis_cache(p)
  > ∇ϕᵢ!(cache, x)
  > cache[3] # Contains the gradient of the basis functions
"""
function ∇ϕᵢ!(cache, x)
  A, b, res = cache
  fill!(res,0.0)
  q = length(res)
  for i=1:q-1
    b[i+1] = i*x^(i-1)
  end
  mul!(res, view(A,:,2:q), view(b,2:q))
end

"""
Function to compute the multiscale basis
"""
function compute_ms_basis!(cache, nc::Int64, q::Int64, p::Int64, D::Function)
  cache_q, cache_p, quad, preallocated_mats = cache
  FULL, FINE, PATCH, BASIS, MATS, ASSEMS, _ = preallocated_mats
  nds_coarse, elem_coarse, _, _, _ = FULL
  nds_fineₛ, elem_fineₛ = FINE
  nds_patchₛ, elem_patchₛ, patch_indices_to_global_indices, _, _ = PATCH
  basis_vec_patch = BASIS
  sKeₛ, sLeₛ, sFeₛ, _ = MATS
  assem_H¹H¹ₛ, assem_H¹L²ₛ, _ = ASSEMS

  for i=1:nc
    gtpi = patch_indices_to_global_indices[i]  
    fillsKe!(sKeₛ[i], cache_q, nds_fineₛ[i], elem_fineₛ[i], q, quad, D)
    fillsLe!(sLeₛ[i],(cache_q, cache_p), nds_fineₛ[i], nds_patchₛ[i], elem_fineₛ[i], elem_patchₛ[i], (q,p), quad)    
    for j=1:p+1
      fillsFe!(sFeₛ[i], cache_p, nds_patchₛ[i], elem_patchₛ[i], p, quad, y->fₖ!(cache_p, y, j, nds_coarse, elem_coarse, i))
      KK = sparse(vec(assem_H¹H¹ₛ[i][1]), vec(assem_H¹H¹ₛ[i][2]), vec(sKeₛ[i]))
      LL = sparse(vec(assem_H¹L²ₛ[i][1]), vec(assem_H¹L²ₛ[i][2]), vec(sLeₛ[i]))
      FF = collect(sparsevec(vec(assem_H¹L²ₛ[i][3]), vec(sFeₛ[i])))
      nfᵢ = size(KK,1)
      fn = 2:nfᵢ-1
      K = KK[fn,fn]; L = LL[fn,:]; F = FF;
      LHS = [K L; L' spzeros(Float64,size(L,2),size(L,2))]
      dropzeros!(LHS)
      RHS = vcat(zeros(Float64,size(K,1)), F)
      RHS = LHS\RHS            
      set_local_basis!(basis_vec_patch, i, gtpi[fn], j, RHS[1:size(K,1)])
    end
  end
end