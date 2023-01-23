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

# Define the assembler for the MS-space
struct MultiScaleSpace <: Strategy end
function MatrixAssembler(x::MultiScaleSpace, fespace::Int64, elem::Matrix{Int64}, l::Int64)
  ndofs = ((fespace+1)*(2l+1) < size(elem,1)*(fespace+1)) ? (fespace+1)*(2l+1) : size(elem,1)*(fespace+1)
  new_elem = _new_elem_matrices(elem, fespace, l, x)
  iM, jM = _get_assembler_matrix(new_elem, ndofs-1)
  iV = Array{Float64}(undef,size(iM))
  fill!(iV,0.0)
  MatrixAssembler(iM, jM, iV)
end 
function VectorAssembler(x::MultiScaleSpace, fespace::Int64, elem::Matrix{Int64}, l::Int64)
  ndofs = ((fespace+1)*(2l+1) < size(elem,1)*(fespace+1)) ? (fespace+1)*(2l+1) : size(elem,1)*(fespace+1)
  new_elem = _new_elem_matrices(elem, fespace, l, x)
  iV = _get_assembler_vector(new_elem, (ndofs-1))
  vV = Array{Float64}(undef,size(iV))
  VectorAssembler(iV, vV)
end 
function _new_elem_matrices(elem, fespace, l, ::MultiScaleSpace)
  N = size(elem,1)
  p = fespace
  l2elems = _new_elem_matrices(elem, p, L²ConformingSpace())
  ndofs = ((p+1)*(2l+1) < (p+1)*N) ? (p+1)*(2l+1) : (p+1)*N
  elems = Matrix{Int64}(undef, N, ndofs)
  fill!(elems,0)
  for el=1:N
    start = (el-l)<1 ? 1 : el-l; last = start+2l
    last = (last>N) ? N : last; start = last-2l
    start = (start ≤ 0) ? 1 : start
    last = (last ≥ N) ? N : last
    elems[el,:] = l2elems[start,1]:l2elems[last,p+1]   
  end
  elems
end
# @btime new_elem = _new_elem_matrices(Ω.elems, p, l, MultiScaleSpace())
# @btime MSₐ = MatrixAssembler(MultiScaleSpace(), p, Ω.elems, l)

# Complete the definition of the multiscale space.
function MultiScale(trian::T, A::Function, fespace::Tuple{Int,Int}, l::Int64, dNodes::Vector{Int64}; Nfine=100, qorder=3) where T<:MeshType
  nel = size(trian.elems,1)
  patch = (2l+1 ≥ nel) ? trian : trian[1:2l+1]
  patch_mesh = 𝒯((patch.nds[1], patch.nds[end]), Nfine)
  q,p = fespace
  new_elems = _new_elem_matrices(trian.elems, p, l, MultiScaleSpace())
  Kₐ = MatrixAssembler(H¹ConformingSpace(), q, patch_mesh.elems)
  Lₐ = MatrixAssembler(H¹ConformingSpace(), L²ConformingSpace(), (q,p), (patch_mesh.elems, patch.elems))
  Fₐ = VectorAssembler(L²ConformingSpace(), p, patch.elems)  
  Rₛ = Matrix{Rˡₕ}(undef,p+1,nel)
  compute_basis_functions!(Rₛ, trian, A, fespace, [Kₐ,Lₐ], [Fₐ]; qorder=qorder, Nfine=Nfine)
  bgSpace = L²Conforming(trian, p)
  nodes = bgSpace.nodes
  MultiScale(trian, l, bgSpace, Rₛ, nodes, dNodes, new_elems)
end 

# Compute and store all the basis functions
# @btime Vₕᴹˢ = MultiScale(Ω, A, (q,p), l, [1,(p+1)*n]; Nfine=nₚ); 

# Function to assemble the matrices corresponding to the multiscale space is similar to the one
# given in fespace.jl (line 11)
function assemble_matrix(U::T, assem::MatrixAssembler, A::Function, M::Function; qorder=10, num_neighbours=2) where T<:MultiScale
  trian = get_trian(U)
  nodes = trian.nds
  els = trian.elems
  p = U.bgSpace.p
  l = U.l  
  new_els = _new_elem_matrices(els, p, l, MultiScaleSpace())
  qs,ws = gausslegendre(qorder)
  i,j,sKe = assem.iM, assem.jM, assem. vM
  sMe = similar(sKe)
  fill!(sKe,0.0); fill!(sMe,0.0)
  nel = size(els,1)
  # Initialize the element-wise local matrices
  ndofs = ((p+1)*(2l+1) < (p+1)*nel) ? (p+1)*(2l+1) : (p+1)*nel
  Me = Array{Float64}(undef,ndofs,ndofs)
  Ke = Array{Float64}(undef,ndofs,ndofs)
  vecBasis = U.basis # Arrange element-wise functions into a vector
    # Do the assembly
    for t=1:nel
      cs = nodes[els[t,:],:]
      b_inds = new_els[t,:]
      hlocal = cs[2]-cs[1]
      fill!(Me,0); fill!(Ke,0.0)
      for k=1:lastindex(qs)
        x = (cs[2]+cs[1])*0.5 .+ 0.5*hlocal*qs[k]
        ϕᵢ = [Λ̃ˡₚ(x, vecBasis[i], vecBasis[i].U; num_neighbours=num_neighbours) for i in b_inds]
        ∇ϕᵢ = [∇Λ̃ˡₚ(x, vecBasis[i], vecBasis[i].U; num_neighbours=num_neighbours) for i in b_inds]
        _local_matrix!(Me, M(x)*ws[k].*(ϕᵢ,ϕᵢ), hlocal, (ndofs-1,ndofs-1))
        _local_matrix!(Ke, A(x)*ws[k].*(∇ϕᵢ,∇ϕᵢ), hlocal, (ndofs-1,ndofs-1))
      end
      for tj=1:ndofs, ti=1:ndofs
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
function assemble_vector(U::T, assem::VectorAssembler, f::Function; qorder=10,num_neighbours=2) where T<:MultiScale
  trian = get_trian(U)
  nodes = trian.nds
  els = trian.elems
  p = U.bgSpace.p
  l = U.l
  new_els = _new_elem_matrices(els, p, l, MultiScaleSpace())
  qs,ws = gausslegendre(qorder)
  k,sFe = assem.iV, assem.vV
  fill!(sFe,0.0)
  nel = size(els,1)
  # Initializa the elemtn-wise vector
  ndofs = ((p+1)*(2l+1) < (p+1)*nel) ? (p+1)*(2l+1) : (p+1)*nel
  Fe = Vector{Float64}(undef,ndofs)
  fill!(Fe,0.0)
  vecBasis = vec(U.basis) # Arrange element-wise functions into a vector
  # Do the assembly
  for t=1:nel
    cs = nodes[els[t,:],:]
    b_inds = new_els[t,:]
    hlocal = cs[2]-cs[1]
    fill!(Fe,0.0)
    for k=1:lastindex(qs)
      x = (cs[2]+cs[1])*0.5 .+ 0.5*hlocal*qs[k]
      ϕᵢ = [Λ̃ˡₚ(x, vecBasis[i], vecBasis[i].U; num_neighbours=num_neighbours) for i in b_inds]
      _local_vector!(Fe, f(x)*ws[k]*ϕᵢ, hlocal, ndofs-1)
    end
    for ti=1:ndofs
      sFe[t,ti] = Fe[ti]
    end 
  end 
  F = collect(sparsevec(vec(k),vec(sFe)))
end

function uₘₛ(x::Float64, sol::Vector{Float64}, U::T; num_neighbours=2) where T<:MultiScale
  Ω = U.trian
  elem = Ω.elems
  new_els = U.new_elem
  nds = Ω.nds
  nel = size(elem,1)
  tree = Ω.tree
  idx, = knn(tree,[x], num_neighbours)
  elem_indx = -1
  for i in idx
    (i ≥ nel) && continue # Finds last point
    interval = nds[elem[i,:]]
    difference = interval .- x
    (difference[1]*difference[2] ≤ 0) ? begin elem_indx = i; break; end : continue
  end
  (elem_indx == -1) && return 0
  uh = sol[new_els[elem_indx,:]]
  b_inds = new_els[elem_indx,:]
  vecBasis = vec(U.basis)
  ϕᵢ = map(i->Λ̃ˡₚ(x, vecBasis[i], vecBasis[i].U; num_neighbours=num_neighbours), b_inds)
  res = dot(uh, ϕᵢ)
  res
end 

ε = 2^-6
A(x) = @. (2 + cos(2π*x/ε))^(-1)
f(x) = @. 1
#u(x) = @. 0.5*x*(1-x)
# Problem parameters
p = 1
q = 1
l = 2
n = 2^3
nₚ = 2^9
# Discretize the domain
Ω = 𝒯((0,1),n)
# Build the Multiscale space. Contains the basis functions in the global sense
Vₕᴹˢ = MultiScale(Ω, A, (q,p), l, [1,n*p+n-p]; Nfine=nₚ, qorder=2)
# Plot the basis function
el = 1
Rₛ = Vₕᴹˢ.basis
plt2 = plot()
for k=1:p+1
  R = Rₛ[k,el]
  plot!(plt2, R.nds, R.Λ, lw=2, label="Basis "*string(k))
  xlims!(plt2,(0,1))
end
# Build the assembler
MSₐ = MatrixAssembler(MultiScaleSpace(), p, Ω.elems, l)
MSₗ = VectorAssembler(MultiScaleSpace(), p, Ω.elems, l)
# Compute the full stiffness and mass matrices
Mₘₛ,Kₘₛ = assemble_matrix(Vₕᴹˢ, MSₐ, A, x->1; qorder=2, num_neighbours=4)
Fₘₛ = assemble_vector(Vₕᴹˢ, MSₗ, f; qorder=2, num_neighbours=4)
#--
# Boundary conditions are applied into the basis functions
#--
uh = Kₘₛ\Fₘₛ
xvals = Vₕᴹˢ.nodes
uhxvals =  map(x->uₘₛ(x, uh, Vₕᴹˢ; num_neighbours=4), xvals)
# uxvals = u.(xvals)
plt = plot(xvals, uhxvals)
# plot!(plt, xvals, uxvals)