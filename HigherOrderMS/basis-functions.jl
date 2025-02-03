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
  cache[j]*sqrt((2*(j-1)+1)/(b-a))
end


function _d(nds::NTuple{2,Float64}, p)
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
    RHS[i,i] = 1.0
  end  
  LHS\RHS
end

"""
Bubble function bₖ,ⱼ ⊆ H¹(Ω) obtained from the Legendre polynomial Λₖ,ⱼ ⊆ L²(Ω)
"""
function bⱼ(x, nds::NTuple{2, Float64}, d, j)
  a, b = nds  
  θ(x) = (b - x)/(b - a)*(x - a)/(b - a)
  res = 0.0
  npolys = size(d,1)  
  for i=1:npolys
    res += d[i,j]*θ(x)*Λₖ!(x, nds, npolys-1, i)
  end
  res
end

"""
ιₖ function used in the construction of the extended bubble function
"""
function ιⱼ(x, nds::NTuple{2,Float64}, nds_patch::NTuple{2,Float64})
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

function _c(domain, nc, p)
  n = ceil(Int64, 0.5*(2*(2p+2)+1))
  x̂, w = gausslegendre(n);
  elem_coarse = [i+j for i=1:nc, j=0:1]
  nds_coarse = LinRange(domain..., nc+1)
  C = [[[zeros(p+1), zeros(p+1)]]; fill([zeros(p+1), zeros(p+1), zeros(p+1)], nc-2); [[zeros(p+1), zeros(p+1)]]]
  d = _d(Tuple(nds_coarse[elem_coarse[1,:]]), p)
  for t=1:nc    
    tri = Tuple(nds_coarse[elem_coarse[t,:]])
    start = max(1,t-1); last = min(nc,t+1)    
    if(t==1 || t==nc) 
      patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
    else
      patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[t,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
    end  
    P = (patch[1][1], patch[end][2]);  
    for g = 1:lastindex(patch)
      G = patch[g]
      x₀, x₁ = G
      xqs = (x₀+x₁)*0.5 .+ (x₁-x₀)*0.5*x̂  
      for i=1:p+1            
        𝐈 = (bⱼ.(xqs, Ref(tri), Ref(d), 1) - ιⱼ.(xqs, Ref(tri), Ref(P))).*Λₖ!.(xqs, Ref(G), p, i)*(x₁-x₀)*0.5
        C[t][g][i] = sum(w.*𝐈)
      end
    end
  end  
  C, elem_coarse, nds_coarse, d
end

"""
νₖ function used in the construction of the extended bubble function
"""
function νⱼ(x, t, CC)
  C, elem_coarse, nds_coarse, d = CC  
  nc = size(elem_coarse,1)
  npolys = size(d,1)  
  start = max(1,t-1); last = min(nc,t+1)
  if(t==1 || t==nc) 
    patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
  else
    patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[t,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
  end
  res = 0.0
  for g = 1:lastindex(C[t])
    G = patch[g]
    for i=1:npolys
      res += C[t][g][i]*bⱼ(x, G, d, i)
    end
  end
  res
end

"""
The extended bubble function Pₕbⱼ = ιₖ + νₖ
"""
function Pₕbⱼ(x, t, CC, α, β)
  _, elem_coarse, nds_coarse, _ = CC
  nc = size(elem_coarse,1)
  tri = Tuple(nds_coarse[elem_coarse[t,:]]) 
  start = max(1,t-1); last = min(nc,t+1)
  if(t==1 || t==nc) 
    patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
  else
    patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[t,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
  end
  P = (patch[1][1], patch[end][2]);
  α*ιⱼ(x, tri, P) + β*νⱼ(x, t, CC)
end
