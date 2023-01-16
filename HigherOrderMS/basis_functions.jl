###################################################################################
# Functions to generate the basis functions for the functions spaces Vₕ and Vₕᵖ(K)
# Here:
# 1) Vₕ is the fine scale space (order q) - Lagrange basis functions
# 2) Vₕᵖ(K) is the L² space (order p) - Shifted Legendre polynomials
###################################################################################
"""
Bₖ is the Legendre polynomial with support K=(a,b)
"""
function Bₖ(x,p,nds)
  # nds is the coordinates of the element
  a,b=nds
  x̂ = -(a+b)/(b-a) + 2/(b-a)*x
  (a < x < b) ? ψ̂(x̂,p) : zeros(Float64,p+1)
end
"""
Returns the projection of Bₖ on H¹₀(D): RˡₕBₖ
"""
function compute_ms_basis(nodes::AbstractVector{Float64}, els::Matrix{Int64},
                          A::Function, fespace, l; Nfine=200, qorder=10)
  # Compute all the basis
  q,p=fespace
  nel = size(els,1)
  nel = size(els,1)
  RˡₕBₖ = Matrix{Rˡₕ}(undef,nel,p+1)
  for k=1:nel
    elcoords = (nodes[els[k,1]],nodes[els[k,2]])
    start = (k-l)>0 ? k-l : 1
    last = (k+l)<nel ? k+l : nel
    for i=1:p+1
      Λₖ(y) = Bₖ(y, p, elcoords)[i]
      new_nodes = nodes[els[start,1]:els[last,2]]
      new_elems = els[start:last,:]
      new_elems = new_elems .- (minimum(new_elems)-1)
      RˡₕBₖ[k,i] = Rˡₕ(Λₖ, A,
                       new_nodes, new_elems;
                       fespace=fespace, N=Nfine, qorder=qorder)
    end
  end
  RˡₕBₖ
end
########################################################################################################################################
