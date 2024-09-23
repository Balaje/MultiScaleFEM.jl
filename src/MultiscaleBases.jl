module MultiscaleBases
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
# File containing the code to extract the 2d patch and compute the multiscale basis functions  
# Contains mainly two parts:
# 1) Some essential functions to extract the patch information
# 2) Goto Line 154: The main routines:
#     (-) MultiScaleTriangulation: Contains the patch information used to compute the multiscale 
#         bases
#     (-) MultiScaleFESpace: Contains the background FESpace and the new multiscale bases matrix
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

#################### BEGIN SOME ESSENTIAL FUNCTIONS THAT ARE USED LATER ON ######################

using Gridap
using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using NearestNeighbors

# For array operations
using SparseArrays
using StaticArrays
using SplitApplyCombine
using LinearAlgebra
using LazyArrays
using FillArrays
using ProgressMeter
using MPI
using FastGaussQuadrature

# Import functions from other modules
using MultiscaleFEM.CoarseToFine: coarsen, get_fine_nodes_in_coarse_elems
using MultiscaleFEM.Assemblers: assemble_loadvec, assemble_stima, poly_exps, Λₖ

"""
Just repeats x n times.
"""
lazy_fill(x, n) = Fill(x, n)

"""
Metric to find the neighbours of the element. Used in BruteTree
"""
struct ElemDist <: NearestNeighbors.Metric end
function NearestNeighbors.Distances.evaluate(::ElemDist, x::AbstractVector, y::AbstractVector)
  dist = abs(x[1] - y[1])
  for i=1:lastindex(x), j=1:lastindex(y)
    dist = min(dist, abs(x[i]-y[j]))
  end
  dist+1
end

"""
Function to obtain the indices of the coarse-scale elements present in the patch of the given element "el".
Uses a KDTree search to find out the Nearest Neighbors.
"""
function get_patch_coarse_elem(Ω::Triangulation, tree::BruteTree, l::Int64, el::Int64)  
  σ = vec(get_cell_node_ids(Ω))
  el_inds = inrange(tree, σ[el], 1) # Find patch of size 1
  for _=2:l # Recursively do this for 2:l and collect the unique indices. 
    X = [inrange(tree, i, 1) for i in σ[el_inds]]
    el_inds = unique(vcat(X...))
  end
  sort(el_inds)
  # There may be a better way to do this... Need to check.
end

"""
Function that returns the indices of the fine-scale elements present inside the patch of the coarse scale elements.
"""
function get_patch_fine_elems(patch_coarse_elems, coarse_to_fine_elem) 
  X = lazy_map(Broadcasting(Reindex(coarse_to_fine_elem)), patch_coarse_elems)
  collect_all_elems(X) = vec(combinedims(X))
  lazy_map(collect_all_elems, X)
end

"""
Function to obtain the global node (in the fine scale) indices of each patch.
Takes the indices of the fine scale elements in the patch and applies the connectivity matrix of the fine scale elements.
"""
get_patch_global_node_ids(patch_fine_elems, σ_fine) = lazy_map(Broadcasting(Reindex(σ_fine)), patch_fine_elems)

######################### END OF ESSENTIAL FUNCTIONS ###############################
################## BEGIN INTERFACE FOR THE MULTISCALE BASES COMPUTATION ##################

struct CoarseTriangulation
  trian::Triangulation
  tree::NNTree
  patch_size::Int64
end
function CoarseTriangulation(domain::Tuple, nc::Int64, l::Int64)
  model_coarse = CartesianDiscreteModel(domain, (nc,nc))
  Ω_coarse = Triangulation(model_coarse)
  σ_coarse = vec(get_cell_node_ids(Ω_coarse))
  # Store the tree of the coarse mesh for obtaining the patch
  R = vec(map(x->SVector(Tuple(x)), σ_coarse))
  tree = BruteTree(R, ElemDist())
  CoarseTriangulation(Ω_coarse, tree, l)
end

struct FineTriangulation
  trian::Triangulation
end
function FineTriangulation(domain::Tuple, nf::Int64)
  model = CartesianDiscreteModel(domain, (nf,nf));
  Ω_fine = Triangulation(model);
  FineTriangulation(Ω_fine)
end


struct MultiScaleTriangulation
  Ωc::CoarseTriangulation
  Ωf::FineTriangulation
  patch_interior_boundary_fine_scale
  patch_local_to_global_map
end
function MultiScaleTriangulation(coarse_trian::CoarseTriangulation, fine_trian::FineTriangulation)
  # Fine Scale Triangulation
  Ωf = fine_trian.trian
  σ_fine = get_cell_node_ids(Ωf) |> vec
  num_fine_cells = num_cells(Ωf)
  # Coarse Scale Triangulation
  Ωc = coarse_trian.trian
  tree = coarse_trian.tree # The brute tree of the coarse mesh for obtaining the patch
  num_coarse_cells = num_cells(Ωc)
  l = coarse_trian.patch_size
  # 1) Get the element-wise map between the coarse and fine-scales
  nsteps = (num_fine_cells/num_coarse_cells) |> sqrt |> log2 |> Int64
  coarse_to_fine_elems = coarsen(num_fine_cells, nsteps) |> vec
  
  # Get the coarse elements on the patch
  patch_coarse_elems = BroadcastVector(get_patch_coarse_elem, lazy_fill(Ωc, num_coarse_cells), lazy_fill(tree, num_coarse_cells), lazy_fill(l, num_coarse_cells), 1:num_coarse_cells)
  
  # Get the elements on the fine scale inside of the coarse scale patch
  patch_fine_elems = get_patch_fine_elems(patch_coarse_elems, coarse_to_fine_elems)
  
  # Get the fine-scale node indices on the patch and separate them as interior and boundary of the patch
  patch_fine_node_ids = get_patch_global_node_ids(patch_fine_elems, σ_fine)
  interior_global = BroadcastVector(get_interior_indices_direct, patch_fine_node_ids)
  boundary_global = BroadcastVector(get_boundary_indices_direct, patch_fine_node_ids)
  
  # Lastly, obtain the global-local map on each elements
  elem_fine = lazy_map(Broadcasting(Reindex(coarse_to_fine_elems)), 1:num_coarse_cells)
  elem_global_node_ids = lazy_map(Broadcasting(Reindex(σ_fine)), elem_fine)
  
  # Return the Object
  MultiScaleTriangulation(coarse_trian, fine_trian, (interior_global, boundary_global), (patch_coarse_elems, elem_global_node_ids))
end

function _compute_reference_L(p::Int64, nc::Int64, nf::Int64)
  ar = Int64(nf/nc)
  domain = (-1.0, 1.0, -1.0, 1.0)
  model = CartesianDiscreteModel(domain, (ar,ar));
  trian = Triangulation(model)
  Vh = TestFESpace(trian, ReferenceFE(lagrangian, Float64, 1), conformity=:H1);
  n_monomials = (p+1)^2
  αβ = poly_exps(p) 
  L = zeros(num_nodes(model), n_monomials)
  for i=1:n_monomials
    b(y) = Λₖ(y, (-1.0,1.0), (-1.0,1.0), p, αβ[i])
    L[:, i] += assemble_loadvec(Vh, b, p+2)
  end
  h = 1/nc
  collect(vec(get_cell_node_ids(trian))), L*(h/2)^2 # To transform to physical coordinates
end

"""
Assemble the inner product between the Legendre Polynomial on the coarse-cell and the Q1-FEM functions on the fine-cells
"""
function assemble_rect_matrix(Ωms::MultiScaleTriangulation, p::Int64)  
  coarse_to_fine_cell_coords = get_coarse_scale_elem_fine_scale_node_indices(Ωms) |> collect;  
  num_coarse_cells = num_cells(Ωms.Ωc.trian)
  num_fine_cells = num_cells(Ωms.Ωf.trian)
  n_monomials = (p+1)^2
  elem_to_dof(x) = n_monomials*x-n_monomials+1:n_monomials*x;
  coarse_dofs = BroadcastVector(elem_to_dof, 1:num_coarse_cells)
  L = spzeros(Float64, num_nodes(Ωms.Ωf.trian), num_coarse_cells*n_monomials) 
  σ_ref, L_ref = _compute_reference_L(p, (num_coarse_cells |> sqrt |> Int64), (num_fine_cells |> sqrt |> Int64))  
  X1 = combinedimsview(combinedims.(coarse_to_fine_cell_coords));
  Z1 = combinedimsview(collect(coarse_dofs))
  Z = combinedimsview(σ_ref)  
  for i=1:num_coarse_cells
    L[X1[:,:,i], Z1[:,i]] .= L_ref[Z,:]
  end  
  L
end

"""
Function to assemble the inner product between the Legendre polynomials on the coarse mesh
"""
function assemble_lm_l2_matrix(Ωms::MultiScaleTriangulation, p::Int64)  
  coarse_trian = Ωms.Ωc.trian  
  n = 2p+1
  xq, wq = gausslegendre(n)
  coarse_cell_coords = get_cell_coordinates(coarse_trian)
  num_coarse_cells = num_cells(coarse_trian)
  n_monomials = (p+1)^2
  l2mat = zeros(Float64,num_coarse_cells*n_monomials,num_coarse_cells*n_monomials)
  index = 1
  αβ = poly_exps(p)
  for t=1:num_coarse_cells
    c = coarse_cell_coords[t]
    nds_x = (c[1][1], c[2][1])
    nds_y = (c[2][2], c[3][2])           
    for q₁=1:lastindex(wq), q₂=1:lastindex(wq)      
      xq1 = (nds_x[2]+nds_x[1])/2 + (nds_x[2]-nds_x[1])/2*xq[q₁]
      yq1 = (nds_y[2]+nds_y[1])/2 + (nds_y[2]-nds_y[1])/2*xq[q₂]
      J = (nds_x[2]-nds_x[1])*0.5*(nds_y[2]-nds_y[1])*0.5
      x1 = Point(xq1, yq1)    
      for i=1:n_monomials
        l2mat[(index-1)+i,(index-1)+i] += wq[q₁]*wq[q₂]*Λₖ(x1,nds_x,nds_y,p,αβ[i])*Λₖ(x1,nds_x,nds_y,p,αβ[i])*J        
      end      
    end
    index = index + n_monomials
  end
  l2mat
end

"""
Function to obtain the interior fine-scale node indices in the coarse-scale patch
"""
get_coarse_scale_patch_fine_scale_interior_node_indices(x::MultiScaleTriangulation) = x.patch_interior_boundary_fine_scale[1]

"""
Function to obtain the boundary fine-scale node indices in the coarse-scale patch
"""
get_coarse_scale_patch_fine_scale_boundary_node_indices(x::MultiScaleTriangulation) = x.patch_interior_boundary_fine_scale[2]

"""
Function to obtain the element indices of the patch of the element
"""
get_coarse_scale_patch_coarse_elem_ids(x::MultiScaleTriangulation) = x.patch_local_to_global_map[1]

"""
Function to obtain the fine-scale node indices present inside each coarse scale elements
"""
get_coarse_scale_elem_fine_scale_node_indices(x::MultiScaleTriangulation) = x.patch_local_to_global_map[2]


struct MultiScaleFESpace <: FESpace
  Ω::MultiScaleTriangulation
  order::Int64
  Uh::FESpace
  basis_vec_ms
  fine_scale_system
end

"""
The multiscale finite element space in 2d.

MultiScaleFESpace(Ωms::MultiScaleTriangulation, q::Int64, p::Int64, A::CellField, qorder::Int64)

INPUT:

- Ωms: MultiScaleTriangulation on which the space needs to be built
- q: Order of Background fine scale discretization
- p: Order of the higher order multiscale method
- A: The diffusion coefficient as a Gridap.CellField
- qorder: Quadrature order 

OUTPUT: MultiScaleFESpace(Ωms, Uh, `basis_vec_ms`)

- Ωms: MultiScaleTriangulation
- Uh: Background fine scale Gridap.FESpace
- `basis_vec_ms`:: Sparse Matrix containing the multiscale bases.

"""
function MultiScaleFESpace(Ωms::MultiScaleTriangulation, p::Int64, Uh::FESpace, fine_scale_matrices)
  # Extract the necessay data from the Triangulation
  Ωc = Ωms.Ωc   
  num_coarse_cells = num_cells(Ωc.trian)
  n_monomials = (p+1)^2
  elem_to_dof(x) = n_monomials*x-n_monomials+1:n_monomials*x;
  coarse_dof_vec = lazy_map(Broadcasting(elem_to_dof), get_coarse_scale_patch_coarse_elem_ids(Ωms));
  coarse_dofs_mat = BroadcastVector(combinedims, coarse_dof_vec)
  coarse_dofs = BroadcastVector(vec, coarse_dofs_mat)  
  legendre_poly = BroadcastVector(elem_to_dof, 1:num_coarse_cells)
  # Since we have zero boundary conditions, we extract only the interior nodes
  patch_interior_fine_scale_dofs = get_coarse_scale_patch_fine_scale_interior_node_indices(Ωms) 
  # Extract the fine scale matrices
  K, L, Λ = fine_scale_matrices
  Ks = lazy_fill(K, num_coarse_cells);
  Ls = lazy_fill(L, num_coarse_cells);
  Λs = lazy_fill(Λ, num_coarse_cells);
  L_size = lazy_fill(size(L), num_coarse_cells);  
  # Compute the basis 
  β = BroadcastVector(get_ms_bases, Ks, Ls, Λs, patch_interior_fine_scale_dofs, coarse_dofs, legendre_poly)
  # Store the basis as Sparse Matrices
  βs = BroadcastVector(build_global_sparse_matrix_from_basis, β, patch_interior_fine_scale_dofs, legendre_poly, L_size)
  # Return the MultiScaleFESpace object
  MultiScaleFESpace(Ωms, p, Uh, βs, (Ks,Ls,Λs))
end

struct MultiScaleCorrections
  Ω::MultiScaleTriangulation
  Vms::MultiScaleFESpace  
  order::Int64  
  ms_corrections
  fine_scale_system
end

"""
Function to obtain the additional corrections for the multiscale basis to improve convergence rates
"""
function MultiScaleCorrections(Vms::MultiScaleFESpace, p::Int64, fine_scale_matrices)
  Ωms = Vms.Ω
  Ωc = Ωms.Ωc 
  q = Vms.order  
  num_coarse_cells = num_cells(Ωc.trian)  
  # For computing the new basis ⊂ Wₚ
  n_monomials_p = (p+1)^2  
  elem_to_dof_p(x) = n_monomials_p*x-n_monomials_p+1:n_monomials_p*x;
  coarse_dof_vec = lazy_map(Broadcasting(elem_to_dof_p), get_coarse_scale_patch_coarse_elem_ids(Ωms));
  coarse_dofs_mat = BroadcastVector(combinedims, coarse_dof_vec)
  coarse_dofs = BroadcastVector(vec, coarse_dofs_mat)  
  # For spanning the new basis of order q
  n_monomials_q = (q+1)^2
  elem_to_dof_q(x) = n_monomials_q*x-n_monomials_q+1:n_monomials_q*x;
  legendre_poly = BroadcastVector(elem_to_dof_q, 1:num_coarse_cells)
  # Global node-ids on patch
  patch_interior_fine_scale_dofs = get_coarse_scale_patch_fine_scale_interior_node_indices(Ωms)
  # Extract the fine-scale matrices
  K, L, M, L₀ = fine_scale_matrices
  Ks = lazy_fill(K, num_coarse_cells)
  Ls = lazy_fill(L, num_coarse_cells)  
  βs = Vms.basis_vec_ms
  Ms = lazy_fill(M, num_coarse_cells) 
  L_size = lazy_fill(size(L₀), num_coarse_cells); 
  # Compute the correction bases using the MultiscaleFESpace
  γ = BroadcastVector(get_ms_bases_corrections, Ks, Ls, Ms, βs, patch_interior_fine_scale_dofs, coarse_dofs, legendre_poly)
  γs = BroadcastVector(build_global_sparse_matrix_from_basis, γ, patch_interior_fine_scale_dofs, legendre_poly, L_size)
  # Return the multiscale object
  MultiScaleCorrections(Ωms, Vms, q, γs, fine_scale_matrices)
end

function build_global_sparse_matrix_from_basis(B::AbstractVecOrMat{Float64}, Ms, Ns, shape::NTuple{2,Int64})
  Is = Ms' .* ones(length(Ns))
  Js = ones(length(Ms))' .* Ns
  sparse(vec(Is), vec(Js), vec(B'), shape...)  
end

"""
Function to obtain the multiscale bases functions
"""
function get_ms_bases(stima::AbstractMatrix{Float64}, lmat::AbstractMatrix{Float64}, rhsmat::AbstractMatrix{Float64}, interior_dofs, coarse_dofs, legendre_poly)
  patch_stima = stima[interior_dofs, interior_dofs]
  patch_lmat = lmat[interior_dofs, coarse_dofs]
  patch_rhs = rhsmat[coarse_dofs, legendre_poly]
  solve_schur_complement(patch_stima, patch_lmat, patch_rhs)
end

"""
Function to obtain the multiscale bases corrections
"""
function get_ms_bases_corrections(stima::AbstractMatrix{Float64}, lmat::AbstractMatrix{Float64}, massma::AbstractMatrix{Float64}, rhsmat::AbstractMatrix{Float64}, interior_dofs, coarse_dofs, legendre_poly)
  patch_stima = stima[interior_dofs, interior_dofs]
  patch_lmat = lmat[interior_dofs, coarse_dofs]
  patch_rhs_1 = massma[interior_dofs, interior_dofs]*rhsmat[interior_dofs, legendre_poly]
  patch_rhs_2 = zeros(length(coarse_dofs), length(legendre_poly))  
  solve_schur_complement(patch_stima, patch_lmat, patch_rhs_1, patch_rhs_2)
end

"""
Function to solve the saddle point system: 
      x = LHS⁻¹RHS
where
      LHS = [K  L; L'  Λ]
      RHS = [f₁; f₂]
"""
function solve_schur_complement(K::AbstractMatrix{Float64}, L::AbstractMatrix{Float64}, f₁::AbstractVecOrMat{Float64}, f₂::AbstractVecOrMat{Float64})
  luK = lu(K)
  L₁ = collect(L)
  F₁ = collect(f₁)
  F₂ = collect(f₂)
  Σ = -L₁'*(luK\L₁);  
  τ = F₂ - L₁'*(luK\F₁)
  Σ⁻¹τ = Σ\τ
  luK\(-L*Σ⁻¹τ + F₁)
end

"""
Functions to obtain the basis functions from the object
"""
get_basis_functions(V::MultiScaleFESpace) = V.basis_vec_ms
get_basis_functions(V::MultiScaleCorrections) = V.ms_corrections

function build_basis_functions!(Bs, Vs, comm::MPI.Comm)
  mpi_size = MPI.Comm_size(comm)
  mpi_rank = MPI.Comm_rank(comm)
  (mpi_rank==0) && println("Using $mpi_size process(es) to compute the solution")
  for (V,B) in zip(Vs, Bs)
    CoarseScale = V.Ω.Ωc
    n_cells_per_proc = ceil(Int64, num_cells(CoarseScale.trian)/mpi_size);  
    @showprogress for i=n_cells_per_proc*(mpi_rank)+1:n_cells_per_proc*(mpi_rank+1)
      B .= B + get_basis_functions(V)[i]    
    end
    BI, BJ, BV = findnz(B);
    if(mpi_size > 1)
      BI = MPI.Gather(BI, comm);
      BJ = MPI.Gather(BJ, comm);
      BV = MPI.Gather(BV, comm);
    end
    if(mpi_rank == 0)      
      B .= sparse(BI, BJ, BV, size(B)...)
    end
  end
  Bs  
end


"""
Function to solve the saddle point system: 
      x = LHS⁻¹RHS
where
      LHS = [K  L; L'  0]
      RHS = [0; f₂]
"""
function solve_schur_complement(K::AbstractMatrix{Float64}, L::AbstractMatrix{Float64}, f₂::AbstractVecOrMat{Float64})
  f₁ = spzeros(Float64, size(K,1), size(f₂,2))
  solve_schur_complement(K, L, f₁, f₂)
end

function get_boundary_indices_direct(σ)
  c = groupcount(combinedims(σ))
  [k for (k,v) in zip(c.indices,c.values) if (v==1) || (v==2) || (v==3)]
end

function get_interior_indices_direct(σ)
  c = groupcount(combinedims(σ))
  [k for (k,v) in zip(c.indices,c.values) if (v==4)]
end

end