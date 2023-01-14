######################################################################
# File containing the definition of the FESpaces used in the problem #
######################################################################
using LinearAlgebra
using ForwardDiff

abstract type FiniteElementSpace <: Any end

"""
mutable struct H¹Conforming <: FiniteElementSpace
  trian::MeshType
  p::Int64
  ϕ̂::Function
  nodes::AbstractVector{Float64}
  elem::Matrix{Int64}
  dirchletNodes::Vector{Int64}
end

- Structure to build the H¹ Conforming space of order p.
- The basis function ϕ̂(x,p) is defined on the reference element (-1,1)
"""
mutable struct H¹Conforming <: FiniteElementSpace
  trian::MeshType
  p::Int64
  ϕ̂::Function
  nodes::AbstractVector{Float64}
  elem::Matrix{Int64}
  dirichletNodes::Vector{Int64}
end
function H¹Conforming(trian::T, p::Int64, dNodes::Vector{Int64}) where T<:MeshType
  N = size(trian.elems,1)
  elem = trian.elems
  elems = Matrix{Int64}(undef,N,p+1)
  for i=1:N
    elems[i,:] = elem[i,1]+(i-1)*(p-1): elem[i,2]+i*(p-1)
  end
  function ϕ̂(x̂)
    xq = LinRange(-1,1,p+1)
    Q = [xq[i]^j for i=1:p+1, j=0:p]
    A = Q\(I(p+1))
    A'*[x̂^i for i=0:p]
  end
  h = trian.H/p
  nodes = trian.nds[1]:h:trian.nds[end]
  H¹Conforming(trian,p,ϕ̂,nodes,elems,dNodes)
end

"""
mutable struct L²Conforming <: FiniteElementSpace
  trian::MeshType
  p::Int64
  Λₖᵖ::Function
  nodes::AbstractVector{Float64}
  elem::Matrix{Int64}
end

- Structure to build the L² Conforming space of order p.
- The basis function Λₖᵖ is defined on the reference element (-1,1)
- The basis functions are Legendre polynomials.
"""
mutable struct L²Conforming <: FiniteElementSpace
  trian::MeshType
  p::Int64
  Λₖᵖ::Function
  nodes::AbstractVector{Float64}
  elem::Matrix{Int64}
end
function L²Conforming(trian::T, p::Int64) where T<:MeshType
  N = size(trian.elems,1)
  elems = Matrix{Int64}(undef,N,p+1)
  elem = trian.elems
  for i=1:N
    elems[i,:] = elem[i,1]+(i-1)*(p): elem[i,2]+i*(p)-1
  end
  function Λₖᵖ(x)
    if (p==0)
      return [1.0]
    elseif(p==1)
      return [1.0, x]
    else
      res = Vector{Float64}(undef, p+1)
      fill!(res,0.)
      res[1] = 1.0
      res[2] = x[1]
      res[3:end] = [(2j-1)/(j)*x*res[j] + (1-j)/(j)*res[j-1] for j=2:p]
      return res
    end
  end
  h = trian.H/p
  nodes = trian.nds[1]:h:trian.nds[end]
  L²Conforming(trian,p,Λₖᵖ,nodes,elems)
end

function get_trian(fespace::T) where T<:FiniteElementSpace
  fespace.trian
end

"""
mutable struct Rˡₕ <: Any
    nds
    els::Matrix{Int64}
    Λ⃗::Vector{Float64}
    λ⃗::Vector{Float64}
end
"""
mutable struct Rˡₕ <: Any
  nds::AbstractVector{Float64}
  els::Matrix{Int64}
  Λ⃗::Vector{Float64}
  λ⃗::Vector{Float64}
end

"""
mutable struct MultiScale <: FiniteElementSpace
  𝒯::MeshType
  bgSpace::L²Conforming
  Λ̃ₖᵖs::Matrix{Rˡₕ}
  elem::Vector{Vector{Int64}}
end
"""
mutable struct MultiScale <: FiniteElementSpace
  trian::MeshType
  bgSpace::L²Conforming
  Λ̃ₖᵖs::Matrix{Rˡₕ}
  elem::Vector{Vector{Int64}}
end

"""
∇(ϕ, x)Function to obtain the gradient of the function ϕ
"""
function ∇(ϕ::Function, x)
  p = length(ϕ(x))-1
  res = Vector{Float64}(undef,p+1)
  fill!(res,0.0)
  for i=1:p+1
    ϕᵢ(y) = ϕ(y)[i]
    res[i] = ForwardDiff.derivative(ϕᵢ,x)
  end
  res
end
