##### ###### ###### ###### ###### ###### ###### ###### #
# Basis functions for the direct and multiscale method #
##### ###### ###### ###### ###### ###### ###### ###### #

function lagrange_basis_cache(p::Int64)
  xq = LinRange(-1,1,p+1)
  Q = [xq[i]^j for i=1:p+1, j=0:p]
  A = Q\I(p+1)
  b = Vector{Float64}(undef,p+1)
  fill!(b,0.0)
  res = similar(b)
  fill!(res,0.0)
  return A', b, res
end
function legendre_basis_cache(p::Int64)
  zeros(Float64,p+1)
end
"""
Function to compute the Lagrange basis functions in (-1,1)
"""
function φᵢ!(cache, x)
  A,b,res = cache
  fill!(res,0.0)
  q = length(res)
  for i=0:q-1
    b[i+1] = x^i      
  end
  mul!(res, A, b)
end
"""
Function to compute the gradient of the Lagrange basis functions in (-1,1)
"""
function ∇φᵢ!(cache, x)
  A,b,res = cache
  fill!(res,0.0)
  q = length(res)
  for i=1:q-1
    b[i+1] = i*x^(i-1)
  end
  mul!(res, view(A,:,2:q), view(b,2:q))
end

"""
Function to compute the Legendre basis functions on (-1,1)
"""
function LP!(cache, x::Float64)
  p = size(cache,1) - 1
  if(p==0)
    cache[1] = 1.0
  elseif(p==1)
    cache[1] = 1.0
    cache[2] = x      
  else
    cache[1] = 1.0
    cache[2] = x
    for j=2:p
      cache[j+1] = (2j-1)/j*x*res[j] - (j-1)/j*res[j-1]
    end
  end
  cache
end  
"""
Shifted Legendre Polynomial with support (a,b)
"""
function Λₖ!(cache, x::Float64, nds::Tuple{Float64,Float64})
  a,b = nds
  fill!(cache,0.0)
  if(a < x < b)
    x̂ = -(b+a)/(b-a) + 2.0*x/(b-a)
    LP!(cache, x̂)
  end
  cache
end

"""
Create cell wise repeating functions for efficient broadcasting
"""
function convert_to_cell_wise(X, nc::Int64)
  tX = typeof(X)
  cell_wise_X = Vector{tX}(undef,nc)
  fill!(cell_wise_X, X)
end