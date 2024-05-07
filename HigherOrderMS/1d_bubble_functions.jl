"""
Bubble function bₖ,ⱼ ⊆ H¹(Ω) obtained from the Legendre polynomial Λₖ,ⱼ ⊆ L²(Ω)
"""
function bⱼ(X, nds::NTuple{2, Float64}, p, order)
  a, b = nds
  n = ceil(Int64, 0.5*(2*(2p+2)+1))
  x̂, w = gausslegendre(n);
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
  end  
  for i=1:npolys
    for q=1:lastindex(w)
      RHS[i,i] += w[q]*(Λₖ!(x[q], nds, p, i)*Λₖ!(x[q], nds, p, i))*h*0.5
    end    
  end
  c = LHS\RHS
  res = 0.0
  for i=1:npolys
    res += c[i,order]*θ(X)*Λₖ!(X, nds, p, i)
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
  res*0.5
end

"""
νₖ function used in the construction of the extended bubble function
"""
function νⱼ(x, tri::NTuple{2,Float64}, patch::NTuple{2,Float64}, p)
  n = ceil(Int64, 0.5*(2*(2p+2)+1))
  x̂, w = gausslegendre(n);
  npolys = (p+1)  
  # Patch
  ã, b̃ = patch  
  # Element
  a, b = tri    
  res = 0.0
  # First element
  if(ã ≈ a)
    N¹K = (a, b), (b, b̃)     
    for G ∈ N¹K
      x₀, x₁ = G
      xqs = (x₀+x₁)/2 .+ (x₁-x₀)/2*x̂            
      for i=1:npolys 
        𝐈 = (bⱼ.(xqs, Ref(tri), p, 1) - ιⱼ.(xqs, Ref(tri), Ref(patch))).*Λₖ!.(xqs, Ref(G), p, i)*(x₁-x₀)*0.5                
        res += sum(w.*𝐈)*bⱼ(x, G, p, i)
      end      
    end
  # Last element
  elseif(b̃ ≈ b)
    N¹K = (ã, a), (a, b)    
    for G ∈ N¹K
      x₀, x₁ = G
      xqs = (x₀+x₁)/2 .+ (x₁-x₀)/2*x̂            
      for i=1:npolys
        𝐈 = (bⱼ.(xqs, Ref(tri), p, 1) - ιⱼ.(xqs, Ref(tri), Ref(patch))).*Λₖ!.(xqs, Ref(G), p, i)*(x₁-x₀)*0.5
        res += sum(w.*𝐈)*bⱼ(x, G, p, i)
      end            
    end    
  # Others
  else    
    N¹K = (ã, a), (a, b), (b, b̃)        
    for G ∈ N¹K
      x₀, x₁ = G
      xqs = (x₀+x₁)/2 .+ (x₁-x₀)/2*x̂              
      for i=1:npolys 
        𝐈 = (bⱼ.(xqs, Ref(tri), p, 1) - ιⱼ.(xqs, Ref(tri), Ref(patch))).*Λₖ!.(xqs, Ref(G), p, i)*(x₁-x₀)*0.5       
        res += sum(w.*𝐈)*bⱼ(x, G, p, i)
      end    
    end
  end    
  res
end

"""
The extended bubble function Pₕbⱼ = ιₖ + νₖ
"""
function Pₕbⱼ(x, nds, patch, p)
  ιⱼ(x, nds, patch) + νⱼ(x, nds, patch, p)
end

using Test

@testset begin
tri = (0.0,0.1); patch = (0.0, 0.2);
p = 3;
h = tri[2]-tri[1];
Π = zeros(p+1, p+1);
n = ceil(Int64, 0.5*(2*(2p+2)+1));
x̂, w = gausslegendre(n);
xqs = (tri[2]+tri[1])/2 .+ (tri[2]-tri[1])/2*x̂  
for i=1:p+1
  for q=1:lastindex(w)
    Π[i,i] += w[q]*(Λₖ!(xqs[q], tri, p, i)*Λₖ!(xqs[q], tri, p, i))*h*0.5
  end  
end
F = zeros(p+1);
xqs = (tri[2]+tri[1])*0.5 .+ (tri[2]-tri[1])*0.5*x̂
for i=1:p+1
  F[i] = 0.0
  for q=1:lastindex(w)
    F[i] += w[q]*(Pb(xqs[q], tri, patch, p))*Λₖ!(xqs[q], tri, p, i)*h*0.5
  end
end

X = Π\F

function E1(p) 
  res = zeros(p+1)
  res[1] = 1.0
  res
end
@test X ≈ E1(p)

# xvals_tri = LinRange(tri..., 50);
# xvals_patch = LinRange(patch..., 241);
# plt1 = Plots.plot(xvals_patch, Λₖ!.(xvals_patch, Ref(tri), Ref(p), Ref(1)), label="Legendre Polynomial \$ \\Lambda_{1,K} \$ ")
# plt2 = Plots.plot(xvals_patch, bⱼ.(xvals_patch, Ref(tri), Ref(p), Ref(1)), label="Bubble function \$ b_{1,K} \$")
# plt3 = Plots.plot(xvals_patch, νⱼ.(xvals_patch, Ref(tri), Ref(patch), Ref(0)), label="\$ \\nu_{K} \$")
# plt4 = Plots.plot(xvals_patch, ιⱼ.(xvals_patch, Ref(tri), Ref(patch)), label="\$ \\iota_{K} \$ ")
# plt5 = Plots.plot(xvals_patch, Pb.(xvals_patch, Ref(tri), Ref(patch), Ref(0)), label="\$ \\iota_{K} + \\nu_{K} \$")

end