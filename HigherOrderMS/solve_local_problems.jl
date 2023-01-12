###############################################################################
# Functions to compute the multi-scale basis by solving the localized problem.
###############################################################################
"""
mutable struct Rˡₕ <: Any
    nds
    els::Matrix{Int64}
    Λ⃗::Vector{Float64}
    λ⃗::Vector{Float64}
end
"""
mutable struct Rˡₕ <: Any
  nds
  els::Matrix{Int64}
  Λ⃗::Vector{Float64}
  λ⃗::Vector{Float64}
end
"""
Function to solve the local problem on Vₕᵖ(Nˡ(K)). We input:
    (1) Λₖ : The Lagrange basis function on k
    (2) A : The Diffusion coefficient
    (3) xn : Patch of order l (Should be supplied as an interval externally)
    (4) kwargs:
        1) fespace = (q,p). Order of polynomials where q:Fine Space and p:The multiscale method.
        2) N = Number of points in the fine space.
        3) qorder = Quadrature order for the fine-scale problem.
"""
function Rˡₕ(Λₖ::Function, A::Function, xn::Tuple,
             nodes_coarse::AbstractVector{Float64}, elem_coarse::Matrix{Int64};
             fespace=(1,1), N=50, qorder=10)
  q,p=fespace
  # Now solve the problem
  nds, els = mesh(xn, N)
  hlocal = (xn[2]-xn[1])/N
  new_nodes = xn[1]:(hlocal)/q:xn[2]
  nel = size(els,1)
  # Boundary, Interior and Total Nodes
  tn = 1:size(new_nodes,1)
  bn = [1, length(new_nodes)]
  fn = setdiff(tn,bn)
  # Get the assemblers
  assem₁ = get_assembler(els,q) # H¹×H¹ innerproduct
  H¹L² = (els,elem_coarse)
  assem₂ = get_assembler(H¹L²,fespace)
  assemble_matrix_vector_H¹_Vₕᵖ(assem₂, (nodes_coarse, nds), (elem_coarse,els),
                                x-> 1, fespace; qorder=qorder)
  # Rˡₕ(nds,
  #     els,
  #     vcat(0,Λ⃗,0),
  #     λ⃗)
end
Base.show(io::IO, z::Rˡₕ) = print(io, "Local basis Rˡₕ on [",z.nds[1],",",z.nds[end],"], ")
