###################
### File eg2.jl ###
###################

# Include all the files.
include("meshes.jl");
include("assemblers.jl");
include("fespaces.jl");
include("basis_functions.jl")
include("local_matrix_vector.jl")
include("assemble_matrices.jl")

A(x) = @. 1; # Diffusion coefficient
n = 6; Nfine = 100; # Coarse and fine mesh size.
p = 1 # Polynomial orders for  L²
q = 1 # Polynomial orders for H¹

Ω = 𝒯((0,1),n); # The full coarse mesh.

start=2; last=4;
NˡK = Ω[start:last];# The submesh from element=start to element=last
VₕᵖNˡK = L²Conforming(NˡK, p); # The L²Conforming space on the coarse mesh

Ωₚ = (NˡK.nds[1], NˡK.nds[end]); # The submesh end points (defines the domain).
NˡKₕ = 𝒯(Ωₚ, Nfine); # Construct the mesh on the patch.
H¹₀NˡK = H¹Conforming(NˡKₕ ,q, [1,(q*Nfine+1)]); # The H¹Conforming space on the fine mesh with Dirichlet boundary conditions

Kₐ = MatrixAssembler(H¹ConformingSpace(), q, NˡKₕ.elems) # Construct the assembler for a(RˡₕΛₖ,v)
Lₐ = MatrixAssembler(H¹ConformingSpace(), L²ConformingSpace(), (q,p), (NˡKₕ.elems, NˡK.elems)) # Construct the assembler for (v,λ)
Fₐ = VectorAssembler(L²ConformingSpace(), p, NˡK.elems) # Construct the vector assembler for (Λₖ,μ)

# Bₖ is the Legendre polynomial with support K=(a,b)
function Bₖ(x,nds,V)
  a,b=nds
  x̂ = -(a+b)/(b-a) + 2/(b-a)*x
  (a ≤ x ≤ b) ? V.basis(x̂) : zeros(Float64,V.p+1)
end

el = 3 # Element Index
local_basis = 1 # Local Basis Index 1:p+1
elem = Ω.nds[Ω.elems[el,1]:Ω.elems[el,2]]; # Get the nodes of element 3.

# Solve the saddle point problem. (Found in fespaces.jl, line 100)
RˡₕΛₖ = Rˡₕ(x->Bₖ(x,elem,VₕᵖNˡK)[local_basis], A, (H¹₀NˡK, VₕᵖNˡK), [Kₐ,Lₐ], [Fₐ]; qorder=4);
# @btime Rˡₕ($(x->Bₖ(x,elem,VₕᵖNˡK)[local_basis]), $A, $(H¹₀NˡK, VₕᵖNˡK), $[Kₐ,Lₐ], $[Fₐ]; qorder=$4);

using Plots
plt = plot(RˡₕΛₖ.nds, RˡₕΛₖ.Λ, label="Basis 2 on element 3", lc=:blue, lw=2)

# Legendre Polynomials basis at the FE nodes
LP = map(y->Bₖ(y,elem,VₕᵖNˡK)[local_basis], RˡₕΛₖ.nds);

plot!(plt, RˡₕΛₖ.nds, LP, label="Legendre Polynomial", lc=:red, lw=2)
plot!(plt, elem[1]:0.01:elem[2], 0*(elem[1]:0.01:elem[2]), label="Element 3", lc=:black, lw=4)
xlims!(plt,(0,1))

## Verify the projection of the MultiScale Basis on the element
function Λ̃ˡₚ(x, R::Rˡₕ, V::A) where A <: H¹Conforming
  Ω = V.trian
  p = V.p
  elem = Ω.elems
  nds = Ω.nds
  nel = size(elem,1)
  new_elem = _new_elem_matrices(elem, p, H¹ConformingSpace())
  for i=1:nel
    cs = nds[elem[i,:]]
    uh = R.Λ[new_elem[i,:]]
    if(cs[1] ≤ x ≤ cs[2])
      x̂ = -(cs[1]+cs[2])/(cs[2]-cs[1]) + 2/(cs[2]-cs[1])*x
      return dot(uh,V.basis(x̂))
    else
      continue
    end
  end
end
#plt1 = plot(RˡₕΛₖ.nds, map(y->Λ̃ˡₚ(y, RˡₕΛₖ, H¹₀NˡK), RˡₕΛₖ.nds));
#plot!(plt1, RˡₕΛₖ.nds, RˡₕΛₖ.Λ)

# L² inner-product of (RˡₕΛₖ, μ)
F₀ = assemble_vector(VₕᵖNˡK, Fₐ, y->Λ̃ˡₚ(y, RˡₕΛₖ, H¹₀NˡK); qorder=5)
# L² inner-product of(Λₖ, μ)
F₁ = assemble_vector(VₕᵖNˡK, Fₐ, y->Bₖ(y, elem, VₕᵖNˡK)[local_basis]; qorder=5)
# Check if they are equal in element el??
display(hcat(F₀, F₁))
#= 4.33681e-19   0.0
3.39256e-9    0.0
4.33681e-18   0.0
1.69628e-8    0.0
0.1           0.1
2.51535e-17  -2.25514e-17
-4.33681e-18   0.0
-1.69628e-8    0.0
3.27971e-18   0.0
-3.39256e-9    0.0 =#
