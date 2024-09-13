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

# Import functions from other modules
using MultiscaleFEM.CoarseToFine: coarsen
using MultiscaleFEM.Assemblers: assemble_loadvec, assemble_stima, assemble_rect_matrix, assemble_rhs_matrix

"""
Just repeats x n times.
"""
lazy_fill(x, n) = Gridap.Arrays.Fill(x, n)

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
  nsteps =  (num_fine_cells/num_coarse_cells) |> sqrt |> log2 |> Int64
  coarse_to_fine_elems = coarsen(num_fine_cells, nsteps) |> vec

  # Get the coarse elements on the patch
  patch_coarse_elems = lazy_map(get_patch_coarse_elem, lazy_fill(Ωc, num_coarse_cells), lazy_fill(tree, num_coarse_cells), lazy_fill(l, num_coarse_cells), 1:num_coarse_cells)

  # Get the elements on the fine scale inside of the coarse scale patch
  patch_fine_elems = get_patch_fine_elems(patch_coarse_elems, coarse_to_fine_elems)

  # Get the fine-scale node indices on the patch and separate them as interior and boundary of the patch
  patch_fine_node_ids = get_patch_global_node_ids(patch_fine_elems, σ_fine)
  interior_global = lazy_map(get_interior_indices_direct, patch_fine_node_ids)
  boundary_global = lazy_map(get_boundary_indices_direct, patch_fine_node_ids)

  # Lastly, obtain the global-local map on each elements
  elem_fine = lazy_map(Broadcasting(Reindex(coarse_to_fine_elems)), 1:num_coarse_cells)
  elem_global_node_ids = lazy_map(Broadcasting(Reindex(σ_fine)), elem_fine)

  # Return the Object
  MultiScaleTriangulation(coarse_trian, fine_trian, (interior_global, boundary_global), (patch_coarse_elems, elem_global_node_ids))
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
  coarse_dofs = lazy_map(elem_to_dof, 1:num_coarse_cells)

  # Since we have zero boundary conditions, we extract only the interior nodes
  patch_interior_fine_scale_dofs = get_coarse_scale_patch_fine_scale_interior_node_indices(Ωms) 

  K, L, Λ = fine_scale_matrices

  Ks = lazy_fill(K, num_coarse_cells);
  Ls = lazy_fill(L, num_coarse_cells);
  Λs = lazy_fill(Λ, num_coarse_cells);

  basis_vec_ms = lazy_map(get_ms_bases, Ks, Ls, Λs, patch_interior_fine_scale_dofs, coarse_dofs)

  MultiScaleFESpace(Ωms, Uh, basis_vec_ms, (Ks,Ls,Λs))
end

"""
Function to obtain the multiscale bases functions
"""

function get_ms_bases(stima::SparseMatrixCSC{Float64, Int64}, lmat::SparseMatrixCSC{Float64,Int64}, rhsmat::SparseMatrixCSC{Float64,Int64}, interior_dofs, coarse_dofs)    
  I = interior_dofs
  J = coarse_dofs
  basis_vec_ms = spzeros(size(lmat,1), length(J))    
  patch_stima = stima[I, I]
  patch_lmat = collect(lmat[I, J])
  patch_rhs = collect(rhsmat[J, J])
  RHS = solve_schur_complement(patch_stima, patch_lmat, patch_rhs)
  basis_vec_ms[I,:] = RHS[1:length(I), :]
  basis_vec_ms
end

"""
Function to solve the saddle point system: 
      x = LHS⁻¹RHS
where
      LHS = [K  L; L'  0]
      RHS = [0; Λ]
"""
function solve_schur_complement(K::AbstractMatrix{Float64}, L::AbstractMatrix{Float64}, f₂::AbstractVecOrMat{Float64})
  luK = lu(K)
  Σ = -L'*(luK\L);
  τ = f₂
  Σ⁻¹τ = Σ\τ
  vcat(luK\(-L*Σ⁻¹τ), Σ⁻¹τ)
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