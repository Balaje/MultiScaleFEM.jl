function Λₖ!(res::Vector{Float64}, x::Float64, nds::Vector{Float64}, p::Int64)
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
function ∇ϕᵢ!(cache, x)
  A, b, res = cache
  fill!(res,0.0)
  q = length(res)
  for i=1:q-1
    b[i+1] = i*x^(i-1)
  end
  mul!(res, view(A,:,2:q), view(b,2:q))
end