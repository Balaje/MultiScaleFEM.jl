###########################################
# Program to solve the Multiscale problem.
###########################################

# include("eg1.jl"); # Contains the basis functions
include("meshes.jl");
include("assemblers.jl");
include("fespaces.jl");
include("basis_functions.jl")
include("local_matrix_vector.jl")
include("assemble_matrices.jl")

#A(x) = @. 1
ε = 2^-7
A(x) = @. (2 + cos(2π*x/ε))^-1

p = 1
q = 1
l = 1
n = 6
nₚ = 100
Ω = 𝒯((0,1),n)

# Define the assembler for the MS-space
struct MultiScaleSpace <: Strategy end
function MatrixAssembler(x::MultiScaleSpace, fespace::Int64, elem::Matrix{Int64}, l::Int64)
  new_elem = _new_elem_matrices(elem, fespace, l, x)
  iM, jM = _get_assembler_matrix(new_elem, (fespace+1)*(l+2)-1)
  iV = Array{Float64}(undef,size(iM))
  fill!(iV,0.0)
  MatrixAssembler(iM, jM, iV)
end 
function VectorAssembler(x::MultiScaleSpace, fespace::Int64, elem::Matrix{Int64}, l::Int64)
  new_elem = _new_elem_matrices(elem, fespace, l, x)
  iV = _get_assembler_vector(new_elem, (fespace+1)*(l+2)-1)
  vV = Array{Float64}(undef,size(iV))
  VectorAssembler(iV, vV)
end 
function _new_elem_matrices(elem, fespace, l, ::MultiScaleSpace)
  N = size(elem,1)
  p = fespace
  l2elems = _new_elem_matrices(elem, p, L²ConformingSpace())
  elems = Matrix{Int64}(undef, N, (p+1)*(l+2))
  fill!(elems,0)
  for el=1:N
    start = (el-l)<1 ? 1 : el-l; last = start+2l
    last = (last>n) ? n : last; start = last-2l
    elems[el,:] = l2elems[start,1]:l2elems[last,p+1]   
  end
  elems
end
new_elem = _new_elem_matrices(Ω.elems, p, l, MultiScaleSpace())
MSₐ = MatrixAssembler(MultiScaleSpace(), p, Ω.elems, l)
# @btime new_elem = _new_elem_matrices(Ω.elems, p, l, MultiScaleSpace())
# @btime MSₐ = MatrixAssembler(MultiScaleSpace(), p, Ω.elems, l)

# Complete the definition of the multiscale space.
function MultiScale(trian::T, A::Function, fespace::Tuple{Int,Int}, l::Int64, dNodes::Vector{Int64}; Nfine=100, qorder=3) where T<:MeshType
  patch = trian[1:2l+1]
  patch_mesh = 𝒯((patch.nds[1], patch.nds[end]), Nfine)
  q,p = fespace
  nel = size(trian.elems,1)
  Kₐ = MatrixAssembler(H¹ConformingSpace(), q, patch_mesh.elems)
  Lₐ = MatrixAssembler(H¹ConformingSpace(), L²ConformingSpace(), (q,p), (patch_mesh.elems, patch.elems))
  Fₐ = VectorAssembler(L²ConformingSpace(), p, patch.elems)  
  Rₛ = Matrix{Rˡₕ}(undef,nel,p+1)
  compute_basis_functions!(Rₛ, trian, A, fespace, [Kₐ,Lₐ], [Fₐ]; qorder=qorder, Nfine=Nfine)
  bgSpace = L²Conforming(trian, p)
  nodes = bgSpace.nodes
  MultiScale(trian, bgSpace, Rₛ, nodes, dNodes)
end 

# Compute and store all the basis functions
# @btime Vₕᴹˢ = MultiScale(Ω, A, (q,p), l, [1,(p+1)*n]; Nfine=nₚ); 
Vₕᴹˢ = MultiScale(Ω, A, (q,p), l, [1,(p+1)*n]; Nfine=nₚ); 

# Function to assemble the matrices corresponding to the multiscale space is similar to the one
# given in fespace.jl (line 11)

# Function to assmeble the vector corresponding to the multiscale space is given in fespace.jl (line 51)
# The above two are the same as assembline the H¹Conforming elements
