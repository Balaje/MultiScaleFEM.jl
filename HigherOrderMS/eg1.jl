using Plots

include("meshes.jl");
include("fespaces.jl");
include("assemblers.jl");
include("local_matrix_vector.jl")
include("assemble_matrices.jl")
include("solve_local_problems.jl");

A(x) = @. 1
#ε = 2^-5
#A(x) = @. (2 + cos(2π*x/ε))^-1

p = 1
q = 1
l = 1
nel_patch = 2l+1 
nel_coarse = 5
nel_fine = 100
# Precompute the assembly matrix for the patch, nel = 2*l + 1 using a dummy mesh
mesh_coarse = 𝒯((0,0.6),2l+1) # nel = 2l+1
mesh_fine = 𝒯((0,0.6),nel_fine)
U = L²Conforming(mesh_coarse,p); # Coarse
V = H¹Conforming(mesh_fine,q,[1,nel_fine+1]); # Fine
# Get the assembly strategies
Kₐ = MatrixAssembler(H¹ConformingSpace(), q, V.elem)
Lₐ = MatrixAssembler(H¹ConformingSpace(), L²ConformingSpace(), (q,p), (V.elem, U.elem))
Fₐ = VectorAssembler(L²ConformingSpace(), p, U.elem)

# Now to compute the new basis functions
function Bₖ(x,nds)
    a,b=nds
    x̂ = -(a+b)/(b-a) + 2/(b-a)*x
    (a ≤ x ≤ b) ? U.basis(x̂) : zeros(Float64,U.p+1)
  end
R = Rˡₕ(y->Bₖ(y,(0.2,0.4))[2], A, (V,U), [Kₐ,Lₐ], [Fₐ]; qorder=10)
#R = Rˡₕ(y->1, A, (V,U), [Kₐ,Lₐ], [Fₐ]; qorder=10)
# function compute_ms_basis(patch::A, fine_mesh::B,
#                assem_coarse::MatrixVectorAssembler, assem_fine::MatrixVectorAssembler) where {A<:MeshType, B<:MeshType}
#   els
#   nel = size(els,1)
#   RˡₕBₖ = Matrix{Rˡₕ}(undef,nel,p+1)
#   for k=1:nel
#     elcoords = (nodes[els[k,1]],nodes[els[k,2]])
#     start = (k-l)>0 ? k-l : 1
#     last = (k+l)<nel ? k+l : nel
#     for i=1:p+1
#       Λₖ(y) = Bₖ(y, p, elcoords)[i]
#       new_nodes = nodes[els[start,1]:els[last,2]]
#       new_elems = els[start:last,:]
#       new_elems = new_elems .- (minimum(new_elems)-1)
#       RˡₕBₖ[k,i] = Rˡₕ(Λₖ, A,
#                       new_nodes, new_elems;
#                       fespace=fespace, N=Nfine, qorder=qorder)
#     end
#   end
# RˡₕBₖ
# end
# Rs = Matrix{Rˡₕ}(undef, nel_coarse, p+1) 
# # using BenchmarkTools
# # @btime Rˡₕ(y->Bₖ(y,(0.2,0.4))[2], A, assem₁, assem₂; qorder=3)
# plt = plot(R.nds, R.Λ)
# xlims!(plt, (0,1))
