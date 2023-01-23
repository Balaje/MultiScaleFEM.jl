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

A(x) = @. 1
#ε = 2^-7
#A(x) = @. (2 + cos(2π*x/ε))^-1

f(x) = @. 1

p = 1
q = 1
l = 4
n = 50
nₚ = 100
Ω = 𝒯((0,1),n)

# Define the assembler for the MS-space
struct MultiScaleSpace <: Strategy end
function MatrixAssembler(x::MultiScaleSpace, fespace::Int64, elem::Matrix{Int64}, l::Int64)
  new_elem = _new_elem_matrices(elem, fespace, l, x)
  iM, jM = _get_assembler_matrix(new_elem, (fespace+1)*(2l+1)-1)
  iV = Array{Float64}(undef,size(iM))
  fill!(iV,0.0)
  MatrixAssembler(iM, jM, iV)
end 
function VectorAssembler(x::MultiScaleSpace, fespace::Int64, elem::Matrix{Int64}, l::Int64)
  new_elem = _new_elem_matrices(elem, fespace, l, x)
  iV = _get_assembler_vector(new_elem, (fespace+1)*(2l+1)-1)
  vV = Array{Float64}(undef,size(iV))
  VectorAssembler(iV, vV)
end 
function _new_elem_matrices(elem, fespace, l, ::MultiScaleSpace)
  N = size(elem,1)
  p = fespace
  l2elems = _new_elem_matrices(elem, p, L²ConformingSpace())
  elems = Matrix{Int64}(undef, N, (p+1)*(2l+1))
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
MSₗ = VectorAssembler(MultiScaleSpace(), p, Ω.elems, l)
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
  Rₛ = Matrix{Rˡₕ}(undef,p+1,nel)
  compute_basis_functions!(Rₛ, trian, A, fespace, [Kₐ,Lₐ], [Fₐ]; qorder=qorder, Nfine=Nfine)
  bgSpace = L²Conforming(trian, p)
  nodes = bgSpace.nodes
  MultiScale(trian, l, bgSpace, Rₛ, nodes, dNodes)
end 

# Compute and store all the basis functions
# @btime Vₕᴹˢ = MultiScale(Ω, A, (q,p), l, [1,(p+1)*n]; Nfine=nₚ); 
Vₕᴹˢ = MultiScale(Ω, A, (q,p), l, [1,n*p+n-p]; Nfine=nₚ); 

# Function to assemble the matrices corresponding to the multiscale space is similar to the one
# given in fespace.jl (line 11)
function assemble_matrix(U::T, assem::MatrixAssembler, A::Function; qorder=10) where T<:MultiScale
  trian = get_trian(U)
  nodes = trian.nds
  els = trian.elems
  p = U.bgSpace.p
  l = U.l
  new_els = _new_elem_matrices(els, p, l, MultiScaleSpace())
  quad = gausslegendre(qorder)
  i,j,sKe = assem.iM, assem.jM, assem. vM
  sMe = similar(sKe)
  fill!(sKe,0.0); fill!(sMe,0.0)
  nel = size(els,1)
  # Initializa the element-wise local matrices
  ndofs = (p+1)*(2l+1)
  Me = Array{Float64}(undef,ndofs,ndofs)
  Ke = Array{Float64}(undef,ndofs,ndofs)
  fill!(Me,0.0); fill!(Ke,0.0)
  vecBasis = vec(U.basis) # Arrange element-wise functions into a vector
    # Do the assembly
  for t=1:nel
    cs = nodes[els[t,:],:]
    b_inds = new_els[t,:]
    hlocal = cs[2]-cs[1]
    ϕᵢ(x) = map(i->Λ̃ˡₚ(x, vecBasis[i], vecBasis[i].U), b_inds)
    ∇ϕᵢ(x) = map(i->∇Λ̃ˡₚ(x, vecBasis[i], vecBasis[i].U), b_inds)
    _local_matrix!(Me, cs, (ϕᵢ,ϕᵢ), A, quad, hlocal, (ndofs-1,ndofs-1))
    _local_matrix!(Ke, cs, (∇ϕᵢ,∇ϕᵢ), A, quad, hlocal, (ndofs-1,ndofs-1))
    for ti=1:ndofs, tj=1:ndofs
      sMe[t,ti,tj] = Me[ti,tj]
      sKe[t,ti,tj] = Ke[ti,tj]
    end
  end
  K = sparse(vec(i),vec(j),vec(sKe))
  M = sparse(vec(i),vec(j),vec(sMe))
  droptol!(M,1e-20), droptol!(K,1e-20)
end 

# Function to assmeble the vector corresponding to the multiscale space is similar to the one 
# given in fespace.jl (line 48)
function assemble_vector(U::T, assem::VectorAssembler, f::Function; qorder=10) where T<:MultiScale
  trian = get_trian(U)
  nodes = trian.nds
  els = trian.elems
  p = U.bgSpace.p
  l = U.l
  new_els = _new_elem_matrices(els, p, l, MultiScaleSpace())
  quad = gausslegendre(qorder)
  k,sFe = assem.iV, assem.vV
  fill!(sFe,0.0)
  nel = size(els,1)
  # Initializa the elemtn-wise vector
  ndofs = (p+1)*(2l+1)
  Fe = Vector{Float64}(undef,ndofs)
  fill!(Fe,0.0)
  vecBasis = vec(U.basis) # Arrange element-wise functions into a vector
  # Do the assembly
  for t=1:nel
    cs = nodes[els[t,:],:]
    b_inds = new_els[t,:]
    hlocal = cs[2]-cs[1]
    ϕᵢ(x) = map(i->Λ̃ˡₚ(x, vecBasis[i], vecBasis[i].U), b_inds)
    _local_vector!(Fe, cs, ϕᵢ, f, quad, hlocal, ndofs-1)
    for ti=1:ndofs
      sFe[t,ti] = Fe[ti]
    end 
  end 
  F = collect(sparsevec(vec(k),vec(sFe)))
end

# Compute the full stiffness and mass matrices
Mₘₛ,Kₘₛ = assemble_matrix(Vₕᴹˢ, MSₐ, A)
Fₘₛ = assemble_vector(Vₕᴹˢ, MSₗ, f)

#--
# Begin applying the boundary conditions
#--
tn = 1:(p*n+n)
bn = Vₕᴹˢ.dNodes
# Get the boundary contributitons
ϕᵢ⁰ = map(i->Λ̃ˡₚ(0, Vₕᴹˢ.basis[i,1], Vₕᴹˢ.basis[i,1].U),2:p+1)
ϕᵢᴺ = map(i->Λ̃ˡₚ(1, Vₕᴹˢ.basis[i,n], Vₕᴹˢ.basis[i,n].U),2:p+1)
BCₘₛ⁰ = Kₘₛ[:,bn[1]].*ϕᵢ⁰'
BCₘₛᴺ = Kₘₛ[:,bn[2]].*ϕᵢᴺ'
# Change the LHS matrix
Kₘₛ[:,2:p+1] -= BCₘₛ⁰
Kₘₛ[:,(p*n+n)-p+1:(p*n+n)] -= BCₘₛᴺ
fn = setdiff(tn,bn)
Fₘₛ = Fₘₛ - Kₘₛ[:,bn[1]]*0 - Kₘₛ[:,bn[2]]*0 
# Solve the remaining system of equations
sol = Kₘₛ[fn,fn]\collect(Fₘₛ[fn])
# Put back the boundary values
uh = Vector{Float64}(undef,n*p+n)
uh[fn] = collect(sol)
uh[bn[1]] = 0 - dot(ϕᵢ⁰,uh[2:p+1])
uh[bn[2]] = 0 - dot(ϕᵢᴺ,uh[(p*n+n)-p+1:(p*n+n)])

function uₘₛ(x::Float64, sol::Vector{Float64}, U::T) where T<:MultiScale
  trian = U.trian
  elem = trian.elems
  nel = size(elem,1)
  nds = trian.nds
  p = U.bgSpace.p
  l = U.l
  new_els = _new_elem_matrices(elem, p, l, MultiScaleSpace())
  vecBasis = vec(U.basis)
  for t=1:nel
    cs = nds[elem[t,:],:]
    b_inds = new_els[t,:]
    uh = sol[b_inds]
    if(cs[1] ≤ x ≤ cs[2])
      ϕᵢ = map(i->Λ̃ˡₚ(x, vecBasis[i], vecBasis[i].U), b_inds)
      return dot(uh, ϕᵢ)
    else
      continue
    end 
  end
end 

xvals = Vₕᴹˢ.nodes
uhxvals =  map(x->uₘₛ(x, uh, Vₕᴹˢ), xvals)
plt = plot(xvals, uhxvals)