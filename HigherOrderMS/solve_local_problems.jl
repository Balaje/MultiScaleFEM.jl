###############################################################################
# Functions to compute the multi-scale basis by solving the localized problem.
###############################################################################
using FastGaussQuadrature
using SparseArrays

function Rˡₕ(Λₖ::Function, A::Function,
             assem₁::MatrixVectorAssembler, assem₂::MatrixVectorAssembler;
             qorder=3)
  (U,_),_ = get_fespaces(assem₁)
  (U,V),_ = get_fespaces(assem₂)
  trian = get_trian(U)
  nodes = trian.nds
  elems = trian.elems
  # Collect the free-nodes
  tn = 1:length(nodes)
  bn = U.dirichletNodes
  fn = setdiff(tn,bn)
  # Use the assemblers and assemble the system
  MM, KK = assemble_matrix_H¹_H¹(assem₁, A; qorder=qorder)
  LL = assemble_matrix_H¹_Vₕᵖ(assem₂, x->1; qorder=qorder)
  FF = assemble_vector_Vₕᵖ(assem₂, Λₖ; qorder=qorder)
  K = KK[fn,fn]; M = MM[fn,fn]; L = LL[fn,:]; Lᵀ = L'; F = FF
  A = [K L; Lᵀ spzeros(size(L,2), size(L,2))]
  dropzeros!(A)
  b = Vector{Float64}(undef, length(fn)+(V.p+1)*size(V.elem,1))
  fill!(b,0.0)
  b[length(fn)+1:end] = F
  sol = A\b
  X = sol[1:length(fn)]
  Y = sol[length(fn)+1:end]
  Rˡₕ(nodes, elems, vcat(0,X,0), Y)
end
