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
  if(a ≤ x ≤ b)
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
  return A', b, res
end
function ϕᵢ!(cache, x)
  A, b, res = cache
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
Function to evaluate the multiscale basis and gradient using trees
"""
function basis_cache(elem::AbstractMatrix{Int64}, p::Int64)  
  nf = size(elem,1)
  new_elem = [p*i+j-(p-1) for i=1:nf, j=0:p]  
  elem_indx = -1
  cache = basis_cache(p)
  nds_cache = Vector{Float64}(undef,nf+1)
  fill!(nds_cache,0.0)
  elem, new_elem, elem_indx, nds_cache, cache
end
"""
Value of the basis function at point x. 
  > cache = basis_cache(kdtree, elem::AbstractMatrix, p)
  > Λₖ(cache, x::Float64, uh::AbstractVector{Float64})
"""
function Λₖ(cache, x::Float64, uh::AbstractArray{Float64}, kdtree::KDTree)
  # nds =  [x[1] for x in kdtree.data]  
  elem, new_elem, elem_indx, nds_cache, bases = cache
  copyto!(nds_cache, @views reinterpret(reshape, Float64, kdtree.data))  
  nel = size(elem,1)
  idx, _ = knn(kdtree, [x], 2, true)  
  for i in idx
    (i ≥ nel) && (i=nel)
    interval = view(nds_cache,view(elem,i,:))
    interval = interval .- x
    (interval[1]*interval[2] ≤ 0) ? begin elem_indx = i; break; end : continue
  end
  # (elem_indx == -1) && begin (res = 0.0); return 0.0; end
  sols = view(uh, view(new_elem, elem_indx, :))
  cs = view(nds_cache, view(elem, elem_indx, :))
  x̂ = -(cs[1]+cs[2])/(cs[2]-cs[1]) + 2/(cs[2]-cs[1])*x
  ϕᵢ!(bases, x̂)
  res = dot(bases[3],sols)
end
function ∇Λₖ(cache, x::Float64, uh::AbstractArray{Float64}, kdtree::KDTree)
  # nds =  [x[1] for x in kdtree.data]
  elem, new_elem, elem_indx, nds_cache, bases = cache
  copyto!(nds_cache, @views reinterpret(reshape, Float64, kdtree.data))  
  nel = size(elem,1)
  idx, _ = knn(kdtree, [x], 2, true)
  for i in idx
    (i ≥ nel) && (i=nel)
    interval = view(nds_cache,view(elem,i,:))
    interval = interval .- x
    (interval[1]*interval[2] ≤ 0) ? begin elem_indx = i; break; end : continue
  end
  (elem_indx == -1) && begin (res = 0.0); return 0.0; end
  sols = view(uh, view(new_elem, elem_indx, :))
  cs = view(nds_cache, view(elem, elem_indx, :))
  x̂ = -(cs[1]+cs[2])/(cs[2]-cs[1]) + 2/(cs[2]-cs[1])*x
  ∇ϕᵢ!(bases, x̂)
  res = dot(bases[3],sols)*(2/(cs[2]-cs[1]))
  return res
end

#= function basis_cache(kdtree::KDTree, nds::AbstractVector{Float64},
  elem::AbstractMatrix{Int64}, p::Int64, l::Int64)

  
end =#