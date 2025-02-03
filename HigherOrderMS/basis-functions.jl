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

"""
ιₖ function used in the construction of the extended bubble function
"""
function ιₖ(x, patch, i)
  # a, b = nds
  if(length(patch) == 2)
    (x₀,x₁), (x₁,x₂) = patch
    if(x₀ < x < x₂)
      ϕ = (x₀ ≤ x ≤ x₁)*[0.5*(x-x₀)/(x₁-x₀),0.0] + 
          (x₁ ≤ x ≤ x₂)*[0.0, 0.5*(x₂-x)/(x₂-x₁)]
      return ϕ[i]
    else
      return 0.0
    end
  else
    (x₀, x₁), (x₁, x₂), (x₂, x₃) = patch
    if(x₀ < x < x₃)    
       ϕ = (x₀ ≤ x ≤ x₁)*[0.5*(x-x₀)/(x₁-x₀), 0.0, 0.0, 0.0] + 
           (x₁ ≤ x ≤ x₂)*[0.0, 0.5*(x₂-x)/(x₂-x₁), 0.5*(x-x₁)/(x₂-x₁), 0.0] + 
           (x₂ ≤ x ≤ x₃)*[0.0, 0.0, 0.0, 0.5*(x₃-x)/(x₃-x₂)]
      # ϕ = ((x-x₀)/(x₁-x₀), (x₂-x)/(x₂-x₁), (x-x₁)/(x₂-x₁), (x₃-x)/(x₃-x₂))
      return ϕ[i]
    else
      return 0.0
    end
  end
end

function ιₖ(x, nds::NTuple{2,Float64}, nds_patch::NTuple{2,Float64})
  a, b = nds
  ã, b̃ = nds_patch
  res = 0.0
  if(ã ≈ a)
    if(ã ≤ x ≤ b)
      res = (x - a)/(b - a)
      # return 1.0
    elseif(b ≤ x ≤ b̃)
      res = (b̃ - x)/(b̃ - b)
    end   
  elseif(b̃ ≈ b)
    if(ã ≤ x ≤ a)
      res = (x - ã)/(a - ã)
    elseif(a ≤ x ≤ b̃)
      res = (b̃ - x)/(b̃ - a)
      # return 1.0
    end
  else
    if(a < x < b)
      res = 1.0
    elseif(x <= a)
      res = (x - ã)/(a - ã)
    elseif(x >= b)
      res = (b̃ - x)/(b̃ - b)
    end
  end

  if(x > b̃ || x < ã)
    res = 0.0
  end
  res*0.5
end

"""
Bubble function bₖ,ⱼ ⊆ H¹(Ω) obtained from the Legendre polynomial Λₖ,ⱼ ⊆ L²(Ω)
"""
function bⱼ(x::Float64, nds::NTuple{2, Float64}, d, j)
  a, b = nds  
  θ(x) = (b - x)/(b - a)*(x - a)/(b - a)
  res = 0.0
  npolys = size(d,1)  
  for i=1:npolys
    res += d[i,j]*θ(x)*Λₖ!(x, nds, npolys-1, i)
  end
  res
end

function _d(nds::NTuple{2,T}, p) where T
  n = ceil(Int64, 0.5*(2*(2p+2)+1))
  x̂, w = gausslegendre(n);
  a, b = nds
  x = (b+a)/2 .+ (b-a)/2*x̂
  h = (b-a)
  θ(x) = (b - x)/(b - a)*(x - a)/(b - a)
  npolys = p+1
  LHS = zeros(npolys, npolys)
  RHS = zeros(npolys, npolys)
  for i=1:npolys, j=1:npolys
    for q=1:lastindex(w)      
      LHS[i,j] += w[q]*θ(x[q])*Λₖ!(x[q], nds, p, i)*Λₖ!(x[q], nds, p, j)*h*0.5
    end
    RHS[i,i] = (h/(2*(i-1)+1))
  end  
  LHS\RHS
end

function _c(nc, el, p; T=Float64)  
  if(el == 1)
    C = [T(3/2) -T(1/2); -T(1/6) T(1/6)] 
  elseif(el == nc)    
    # C = [T(3/2) -T(1/2); -T(1/6) T(1/6)]    
    C = [-T(1/2) T(3/2); -T(1/6) T(1/6)]    
  else
    C = [-T(1/2) T(1) -T(1/2); -T(1/6) T(0) T(1/6)]    
  end  
  if(p>=2)
    Z = spzeros(T, p-1, size(C,2))
    C = [C; Z]
  end 
  C[1:p+1,:]
end