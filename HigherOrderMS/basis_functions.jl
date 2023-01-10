############################################################################################
# Functions to generate the basis functions for the functions spaces Vₕ and Vₕᵖ(K)
# Here:
# 1) Vₕ is the fine scale space (order q) - Lagrange basis functions (Handled using Gridap)
# 2) Vₕᵖ(K) is the L² space (order p) - Shifted Legendre polynomials
############################################################################################

"""
Function to generate the Legendre polynomials upto order p.
The polynomials are defined in the reference interval (-1,1)
"""
function Λₖᵖ(x, p::Int64)
  (p==0) && return [1.0]
  (p==1) && return [1.0, x]
  (p > 1) && begin
    res = Vector{Float64}(undef, p+1)
    fill!(res,0.)
    res[1] = 1.0
    res[2] = x[1]
    for j=2:p
      res[j+1] = (2j-1)/(j)*x*res[j] + (1-j)/(j)*res[j-1]
    end
    return res
  end
end
"""
Basis function and gradient for the Lagrange multiplier of order p:
  (This is the L² function and is equal to the Legendre polynomials)
"""
ψ̂(x, p) = Λₖᵖ(x, p)
function ∇ψ̂(x, p)
  (p==0) && return 0.0
  (p==1) && return [0.0, 1.0]
  (p > 1) && begin
    dRes = Vector{Float64}(undef, p+1)
    Res = Vector{Float64}(undef, p+1)
    fill!(dRes,0.); fill!(Res,0.)
    dRes[1] = 0.0;   dRes[2] = 1.0
    Res[1] = 1.0; Res[2] = x
    for j=2:p
      Res[j+1] = (2j-1)/(j)*x*Res[j] + (1-j)/(j)*Res[j-1]
      dRes[j+1] = (2j-1)/(j)*(Res[j] + x*dRes[j]) + (1-j)/(j)*dRes[j-1]
    end
    return dRes
  end
end

"""
Value of the multiscale basis at x:
(1) Accepts the basis FEM solution and returns the value at x
"""
Λ̃ₖˡ(x, R::Rˡₕ) = R.Λ(Point(x))
"""
Gradient of the multiscale bases at x
(1) Accepts the basis FEM solution and returns the value at x
"""
∇Λ̃ₖˡ(x, R::Rˡₕ) = ∇(R.Λ)(Point(x))[1]
"""
Bₖ is the Legendre polynomial with support K=(a,b)
"""
function Bₖ(x,(l,p),nds)
    # nds is the coordinates of the element
    a,b=nds
    x̂ = -(a+b)/(b-a) + 2/(b-a)*x
    (a < x < b) ? ψ̂(x̂,p)[l] : 0.0
end
"""
Returns the projection of Bₖ on H¹₀(D): RˡₕBₖ
"""
function compute_ms_basis(nodes, els, A::Function, fespace, l; Nfine=200, qorder=10)
    # Compute all the basis
    q,p=fespace
    nel = size(els,1)
    RˡₕBₖ = Matrix{Rˡₕ}(undef,nel,p+1)
    for k=1:nel
      elcoords = (nodes[els[k,1]],nodes[els[k,2]])
      start = (k-l)>0 ? k-l : 1
      last = (k+l)<nel ? k+l : nel
      for i=1:p+1
        Λₖ(y) = Bₖ(y[1], (i,p), elcoords)
        domain = (nodes[els[start,1]], nodes[els[last,2]])        
        RˡₕBₖ[k,i] = Rˡₕ(Λₖ, A, domain; fespace=fespace, N=Nfine, qorder=qorder)
      end
    end
    RˡₕBₖ
end