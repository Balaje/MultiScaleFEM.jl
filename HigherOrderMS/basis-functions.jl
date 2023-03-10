##### ###### ###### ###### ###### ###### ####
# Contains the shifted Legendre polynomials #
##### ###### ###### ###### ###### ###### ####
"""
Function to compute the Legendre basis functions on (-1,1)
"""
function LP!(cache::Vector{Float64}, x::Float64)
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
function Λₖ!(x, nds::Tuple{Float64,Float64}, p::Int64, j::Int64)
  a,b = nds
  cache = Vector{Float64}(undef, p+1)
  fill!(cache,0.0)
  if(a < x < b)
    x̂ = -(b+a)/(b-a) + 2.0*x/(b-a)
    LP!(cache, x̂)
  end
  cache[j]
end