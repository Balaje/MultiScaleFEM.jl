###################################################################################
# Functions to generate the basis functions for the functions spaces Vₕ and Vₕᵖ(K)
# Here:
# 1) Vₕ is the fine scale space (order q) - Lagrange basis functions
# 2) Vₕᵖ(K) is the L² space (order p) - Shifted Legendre polynomials
###################################################################################
"""
Bₖ is the Legendre polynomial with support K=(a,b)
"""
# Now to compute the new basis functions on a patch
function Bₖ(x,nds,V)
  a,b=nds
  x̂ = -(a+b)/(b-a) + 2/(b-a)*x
  (a ≤ x ≤ b) ? V.basis(x̂) : zeros(Float64,V.p+1)
end
"""
Returns the projection of Bₖ on H¹₀(D): RˡₕBₖ
"""
function compute_basis_functions!(Rₛ::Matrix{Rˡₕ},
  Ω::T, A::Function, fespace, MatAssems::Vector{MatrixAssembler},
  VecAssems::Vector{VectorAssembler};
  qorder=3, Nfine=100) where T<:MeshType

  q,p = fespace
  n = size(Ω.elems,1)
  Kₐ, Lₐ = MatAssems
  Fₐ, = VecAssems
  for el=1:n
    # Get the start and last index of the patch
    start = (el-l)<1 ? 1 : el-l; last = start+2l
    last = (last>n) ? n : last; start = last-2l
    start = (start ≤ 0) ? 1 : start
    last = (last ≥ n) ? n : last
    NˡK = Ω[start:last]
    Ωₚ = (NˡK.nds[1], NˡK.nds[end])
    elem = Ω.nds[Ω.elems[el,1]:Ω.elems[el,2]]
    NˡKₕ = 𝒯(Ωₚ, Nfine)
    VₕᵖNˡK = L²Conforming(NˡK, p); # Coarse Mesh
    H¹₀NˡK = H¹Conforming(NˡKₕ ,q, [1,(q*Nfine+1)]); # Fine Mesh
    for i=1:p+1
#      display(plot(LinRange(-1,1,800), map(x->Bₖ(x,elem,VₕᵖNˡK)[p+1], LinRange(-1,1,800)), xlims=(0,1)))
#      sleep(0.5)
      R = Rˡₕ(x->Bₖ(x,elem,VₕᵖNˡK)[i], A, x->1.0, (H¹₀NˡK, VₕᵖNˡK), [Kₐ,Lₐ], [Fₐ]; qorder=qorder)
      Rₛ[i,el] = R
    end
  end
  Rₛ
end
