######################################################################
# File containing the definition of the FESpaces used in the problem #
######################################################################
using LinearAlgebra
using ForwardDiff
using FastGaussQuadrature
using SparseArrays

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
  basis::Function
  nodes::AbstractVector{Float64}
  dirichletNodes::Vector{Int64}
end
function H¹Conforming(trian::T, p::Int64, dNodes::Vector{Int64}) where T<:MeshType
  function ϕ̂(x̂)
    xq = LinRange(-1,1,p+1)
    Q = [xq[i]^j for i=1:p+1, j=0:p]
    A = Q\(I(p+1))
    A'*[x̂^i for i=0:p]
  end
  h = trian.H/p
  nodes = trian.nds[1]:h:trian.nds[end]
  H¹Conforming(trian,p,ϕ̂,nodes,dNodes)
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
  basis::Function
  nodes::AbstractVector{Float64}  
end
function L²Conforming(trian::T, p::Int64) where T<:MeshType
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
  L²Conforming(trian,p,Λₖᵖ,nodes)
end

function get_trian(fespace::T) where T<:FiniteElementSpace
  fespace.trian
end

"""
mutable struct Rˡₕ{T<:FiniteElementSpace} <: Any
    nds
    els::Matrix{Int64}
    Λ⃗::Vector{Float64}
    λ⃗::Vector{Float64}
end
"""
mutable struct Rˡₕ{T<:FiniteElementSpace} <: Any
  nds::AbstractVector{Float64}
  Λ::Vector{Float64}
  λ::Vector{Float64}
  U::T
end
function Rˡₕ(Λₖ::Function, A::Function, Us::Tuple{T1,T2}, MatAssems::VecOrMat{MatrixAssembler}, 
    VecAssems::VecOrMat{VectorAssembler}; qorder=3) where {T1<:FiniteElementSpace, T2<:FiniteElementSpace}
  U,V = Us
  Kₐ, Lₐ = MatAssems
  Fₐ, = VecAssems
  nodes = U.nodes
  # Collect the free-nodes
  tn = 1:length(nodes)
  bn = U.dirichletNodes
  fn = setdiff(tn,bn)
  # Use the assemblers and assemble the system
  ~,KK = assemble_matrix(U, Kₐ, A; qorder=qorder)
  LL = assemble_matrix(U, V, Lₐ, x->1; qorder=qorder)
  FF = assemble_vector(V, Fₐ, Λₖ; qorder=qorder)
  K = KK[fn,fn]; L = LL[fn,:]; Lᵀ = L'; F = FF
  A = [K L; Lᵀ spzeros(size(L,2), size(L,2))]
  b = Vector{Float64}(undef, length(fn)+length(F))  
  dropzeros!(A)  
  fill!(b,0.0)
  b[length(fn)+1:end] = F
  sol = A\b
  X = sol[1:length(fn)]
  Y = sol[length(fn)+1:end]
  Rˡₕ(nodes, vcat(0,X,0), Y, U)
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
Value of the multiscale basis at x:
(1) Accepts the basis FEM solution and returns the value at x
"""
function Λ̃ₖˡ(x, R::Rˡₕ)
  nds = R.nds; elem=R.els; uh = R.Λ;
  @assert R.U isa H¹Conforming
  new_elem = R.U.elem  
  nel = size(elem,1)
  for i=1:nel    
    uh_elem = uh[new_elem[i,:]]
    nds_elem = nds[elem[i,:]]
    hl = nds_elem[2]-nds_elem[1]
    if(nds_elem[1] ≤ x ≤ nds_elem[2])
      x̂ = -(nds_elem[2]+nds_elem[1])/hl + (2/hl)*x
      return dot(uh_elem, U.ϕ̂(x̂))
    else
      continue
    end
  end
end
"""
Gradient of the multiscale bases at x
(1) Accepts the basis FEM solution and returns the value at x
"""
function ∇Λ̃ₖˡ(x, R::Rˡₕ)
  nds = R.nds; elem=R.els; uh = R.Λ⃗;
  @assert R.U isa H¹Conforming
  new_elem = R.U.elem  
  nel = size(elem,1)
  for i=1:nel    
    uh_elem = uh[new_elem[i,:]]
    nds_elem = nds[elem[i,:]]
    hl = nds_elem[2]-nds_elem[1]
    if(nds_elem[1] ≤ x ≤ nds_elem[2])
      x̂ = -(nds_elem[2]+nds_elem[1])/hl + (2/hl)*x
      return dot(uh_elem, ∇(ϕ̂,x̂))*(2/hl)
    else
      continue
    end
  end
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
