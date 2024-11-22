##### ###### ###### ###### ###### ###### ####
# Contains the shifted Legendre polynomials #
##### ###### ###### ###### ###### ###### ####
"""
Function to compute the Legendre basis functions on (-1,1)
"""
function LP!(cache::Vector{T}, x::T) where T<:Number
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
      cache[j+1] = (2j-1)/j*x*cache[j] - (j-1)/j*cache[j-1]
    end
  end
  cache
end  
"""
Shifted Legendre Polynomial with support (a,b)
"""
function Λₖ!(x, nds::NTuple{2,T}, p::Int64, j::Int64) where T<:Number
  a,b = nds
  cache = Vector{T}(undef, p+1)
  fill!(cache,0.0)
  if(a < x < b)
    x̂ = -(b+a)/(b-a) + 2.0*x/(b-a)
    LP!(cache, x̂)
  end
  cache[j]
end