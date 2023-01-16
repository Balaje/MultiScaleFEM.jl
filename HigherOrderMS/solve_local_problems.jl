###############################################################################
# Functions to compute the multi-scale basis by solving the localized problem.
###############################################################################
using FastGaussQuadrature
using SparseArrays

function Rˡₕ(Λₖ::Function, A::Function, Us::Tuple{T1,T2}, MatAssems::VecOrMat{MatrixAssembler}, 
             VecAssems::VecOrMat{VectorAssembler}; qorder=3) where {T1<:FiniteElementSpace, T2<:FiniteElementSpace}
  U,V = Us
  Kₐ, Lₐ = MatAssems
  Fₐ, = VecAssems
  trian = get_trian(U)
  nodes = trian.nds
  # Collect the free-nodes
  tn = 1:length(nodes)
  bn = U.dirichletNodes
  fn = setdiff(tn,bn)
  # Use the assemblers and assemble the system
  MM,KK = assemble_matrix(U, Kₐ, A; qorder=qorder)
  LL = assemble_matrix(U, V, Lₐ, A; qorder=qorder)
  # K = KK[fn,fn]; M = MM[fn,fn]; L = LL[fn,:]; Lᵀ = L'; F = FF
  # A = [K L; Lᵀ spzeros(size(L,2), size(L,2))]
  # dropzeros!(A)
  # b = Vector{Float64}(undef, length(fn)+(V.p+1)*size(V.elem,1))
  # fill!(b,0.0)
  # b[length(fn)+1:end] = F
  # sol = A\b
  # X = sol[1:length(fn)]
  # Y = sol[length(fn)+1:end]
  # Rˡₕ(nodes, elems, vcat(0,X,0), Y, U)
end
