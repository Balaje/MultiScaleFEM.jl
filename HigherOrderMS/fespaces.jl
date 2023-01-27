######################################################################
# File containing the definition of the FESpaces used in the problem #
######################################################################
using LinearAlgebra
using ForwardDiff
using FastGaussQuadrature
using SparseArrays

abstract type FiniteElementSpace <: Any end
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
  new_elem::Matrix{Int64}
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
  new_elem = _new_elem_matrices(trian.elems, p, H¹ConformingSpace())
  H¹Conforming(trian,p,ϕ̂,nodes,dNodes,new_elem)
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
  new_elem::Matrix{Int64}
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
      res[2] = x
      for j=2:p
        res[j+1] = (2j-1)/(j)*x*res[j] - (j-1)/(j)*res[j-1]
      end
      return res
    end
  end
  h = trian.H/p
  nodes = trian.nds[1]:h:trian.nds[end]
  new_elem = _new_elem_matrices(trian.elems, p, L²ConformingSpace())
  L²Conforming(trian,p,Λₖᵖ,nodes,new_elem)
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
function Rˡₕ(Λₖ::Function, A::Function, M::Function, Us::Tuple{T1,T2}, MatAssems::VecOrMat{MatrixAssembler},
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
  _,KK = assemble_matrix(U, Kₐ, A, M; qorder=qorder)
  LL = assemble_matrix(U, V, Lₐ, x->1.0; qorder=qorder)
  FF = assemble_vector(V, Fₐ, Λₖ; qorder=qorder)
  @show FF
  K = KK[fn,fn]; L = LL[fn,:]; Lᵀ = L'; F = FF
  A = [K L; Lᵀ spzeros(size(L,2), size(L,2))]
  # b = Vector{Float64}(undef, length(fn)+length(F))
  b = Vector{Float64}(undef, size(A,1))
  dropzeros!(A)
  fill!(b,0.0)
  b[length(fn)+1:end] = F
  sol = A\b
  X = sol[1:length(fn)]
  Y = sol[length(fn)+1:end]
  Rˡₕ(nodes, vcat(0,X,0), Y, U)
end
"""
Value of the multiscale basis at x:
(1) Accepts the basis FEM solution and returns the value at x
"""
function Λ̃ˡₚ(x::Float64, R::Rˡₕ, V::A; num_neighbours=2) where A <: H¹Conforming
  Ω = V.trian
  elem = Ω.elems
  new_elem = V.new_elem
  nds = Ω.nds
  nel = size(elem,1)
  tree = Ω.tree
  idx, = knn(tree, [x], num_neighbours, true)
  elem_indx = -1
  for i in idx
    (i ≥ nel) && (i=nel) # Finds last point
    interval = nds[elem[i,:]]
    difference = interval .- x
    (difference[1]*difference[2] ≤ 0) ? begin elem_indx = i; break; end : continue
  end
  (elem_indx == -1) && return 0.0
  uh = R.Λ[new_elem[elem_indx,:]]
  cs = nds[elem[elem_indx,:]]
  x̂ = -(cs[1]+cs[2])/(cs[2]-cs[1]) + 2/(cs[2]-cs[1])*x
  res = dot(uh,V.basis(x̂))
  res
end
"""
Gradient of the multiscale bases at x
(1) Accepts the basis FEM solution and returns the value at x
"""
function ∇Λ̃ˡₚ(x::Float64, R::Rˡₕ, V::A; num_neighbours=2) where A <: H¹Conforming
  Ω = V.trian
  p = V.p
  elem = Ω.elems
  new_elem = V.new_elem
  nds = Ω.nds
  nel = size(elem,1)
  tree = Ω.tree
  idx, = knn(tree, [x], num_neighbours, true)
  elem_indx = -1
  for i in idx
    (i ≥ nel) && (i=nel) # Finds last point
    interval = nds[elem[i,:]]
    difference = interval .- x
    (difference[1]*difference[2] ≤ 0) ? begin elem_indx = i; break; end : continue
  end
  (elem_indx == -1) && return 0.0
  uh = R.Λ[new_elem[elem_indx,:]]
  cs = nds[elem[elem_indx,:]]
  x̂ = -(cs[1]+cs[2])/(cs[2]-cs[1]) + 2/(cs[2]-cs[1])*x  
  ϕᵢ(x) = V.basis(x)
  ∇ϕᵢ = ∇(ϕᵢ,x̂)*2/(cs[2]-cs[1])
  res = dot(uh, ∇ϕᵢ)
  res
end

######### ######### ######### ######### ######### ######### 
######### Definition of the Multiscale space #### #########
######### ######### ######### ######### ######### ######### 
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
  l::Int64
  bgSpace::L²Conforming
  basis::Matrix{Rˡₕ}
  nodes::AbstractVector{Float64}
  dNodes::Vector{Int64}
  new_elem::Matrix{Int64}
end
"""
Function to build the Multiscale space
"""
function MultiScale(trian::T, A::Function, fespace::Tuple{Int,Int}, l::Int64, dNodes::Vector{Int64}; Nfine=100, qorder=3) where T<:MeshType
  nel = size(trian.elems,1)
  q,p = fespace
  patch = ((2l+1)*(p+1) < nel*(p+1)) ? trian[1:2l+1] : trian
  patch_mesh = 𝒯((patch.nds[1], patch.nds[end]), Nfine)
  new_elems = _new_elem_matrices(trian.elems, p, l, MultiScaleSpace())
  Kₐ = MatrixAssembler(H¹ConformingSpace(), q, patch_mesh.elems)
  Lₐ = MatrixAssembler(H¹ConformingSpace(), L²ConformingSpace(), (q,p), (patch_mesh.elems, patch.elems))
  Fₐ = VectorAssembler(L²ConformingSpace(), p, patch.elems)  
  Rₛ = Matrix{Rˡₕ}(undef,p+1,nel)
  compute_basis_functions!(Rₛ, trian, A, fespace, [Kₐ,Lₐ], [Fₐ], l; qorder=qorder, Nfine=Nfine)
  bgSpace = L²Conforming(trian, p)
  nodes = bgSpace.nodes
  MultiScale(trian, l, bgSpace, Rₛ, nodes, dNodes, new_elems)
end 
"""
Evaluate the multiscale function at a point
"""
function uₘₛ(x::Float64, sol::Vector{Float64}, U::T; num_neighbours=2) where T<:MultiScale
  Ω = U.trian
  elem = Ω.elems
  new_els = U.new_elem
  nds = Ω.nds
  nel = size(elem,1)
  tree = Ω.tree
  idx, = knn(tree, [x], num_neighbours, true)
  elem_indx = -1
  for i in idx
    (i ≥ nel) && (i=nel) # Finds last point
    interval = nds[elem[i,:]]
    difference = interval .- x
    (difference[1]*difference[2] ≤ 0) ? begin elem_indx = i; break; end : continue
  end
  (elem_indx == -1) && return 0
  uh = sol[new_els[elem_indx,:]]
  b_inds = new_els[elem_indx,:]
  ϕᵢ = map(i->Λ̃ˡₚ(x, U.basis[i], U.basis[i].U; num_neighbours=num_neighbours), b_inds)
  res = dot(uh, ϕᵢ)
  res
end 