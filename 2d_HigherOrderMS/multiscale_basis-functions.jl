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
  σ = get_cell_node_ids(Ω)
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

get_unique_node_ids(x) = unique(combinedims(x))

######################### END OF ESSENTIAL FUNCTIONS ###############################
################## BEGIN INTERFACE FOR THE MULTISCALE BASES COMPUTATION ##################

"""
The Combined MultiScale Triangulation

MultiScaleTriangulation(domain::Tuple, nf::Int64, nc::Int64, l::Int64)

INPUT: 

- domain: Tuple containing the end points of the domain
- nf: Number of fine-scale partitions on the axes
- nc: Number of coarse-scale partitions on the axes
- l: Patch size.

OUTPUT:

- Ωc: Gridap.Triangulation containing the coarse triangulation
- Ωf: Gridap-Triangulation containing the fine triangulation
- `patch_models_coarse`: A Vector{Gridap.DiscreteModel} containing the patch mesh in the coarse scale
- `patch_models_fine`: A Vector{Gridap.DiscreteModel} containing the patch mesh in the fine scale
- `interior_boundary_local`: Tuple containing the local (fine-scale) node numbering of (interior, boundary) in the patch meshes 
- `interior_boundary_global`: Tuple containing the global (fine-scale) node numbering of (interior, boundary) in the patch meshes 
- `local_to_global_map`: Global (fine-scale) indices in each coarse element.

"""
struct MultiScaleTriangulation
  Ωc::Triangulation
  Ωf::Triangulation
  interior_boundary_global
  local_to_global_map
end
function MultiScaleTriangulation(domain::Tuple, nf::Int64, nc::Int64, l::Int64)
  # Fine Scale Space
  model_h = CartesianDiscreteModel(domain, (nf,nf))
  Ωf = Triangulation(model_h)
  σ_fine = vec(get_cell_node_ids(Ωf))
  # Coarse Scale Space
  model_H = CartesianDiscreteModel(domain, (nc,nc))
  Ωc = Triangulation(model_H)
  σ_coarse = vec(get_cell_node_ids(Ωc))
  # Store the tree of the coarse mesh for obtaining the patch
  R = vec(map(x->SVector(Tuple(x)), σ_coarse))
  tree = BruteTree(R, ElemDist())
  # Get the data structures required to build the discrete DiscreteModels
  num_coarse_cells = num_cells(Ωc)
  # 1) Get the element indices inside the patch on the coarse and fine scales
  nsteps =  (Int64(log2(nf/nc)))
  coarse_to_fine_elems = vec(coarsen(model_h, nsteps))

  patch_coarse_elems = lazy_map(get_patch_coarse_elem, 
    lazy_fill(Ωc, num_coarse_cells), 
    lazy_fill(tree, num_coarse_cells), 
    lazy_fill(l, num_coarse_cells), 
    1:num_coarse_cells)

  patch_fine_elems = get_patch_fine_elems(patch_coarse_elems, coarse_to_fine_elems)

  patch_fine_node_ids = get_patch_global_node_ids(patch_fine_elems, σ_fine)
  interior_global = lazy_map(get_interior_indices_direct, patch_fine_node_ids)
  boundary_global = lazy_map(get_boundary_indices_direct, patch_fine_node_ids)
  interior_boundary_global = (interior_global, boundary_global)

  # Lastly, obtain the global-local map on each elements
  elem_fine = lazy_map(Broadcasting(Reindex(coarse_to_fine_elems)), 1:num_coarse_cells)
  elem_global_node_ids = lazy_map(Broadcasting(Reindex(σ_fine)), elem_fine)

  # Return the Object
  MultiScaleTriangulation(Ωc, Ωf, interior_boundary_global, (elem_global_node_ids, patch_coarse_elems))
end


struct MultiScaleFESpace <: FESpace
  Ω::MultiScaleTriangulation
  Uh::FESpace
  basis_vec_ms
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
function MultiScaleFESpace(Ωms::MultiScaleTriangulation, q::Int64, p::Int64, A::CellField, qorder::Int64)
  # Extract the necessay data from the Triangulation
  Ωf = Ωms.Ωf
  Ωc = Ωms.Ωc
  elem_global_node_ids = Ωms.local_to_global_map[1]
  elem_global_unique_node_ids = lazy_map(get_unique_node_ids, elem_global_node_ids)
  # Build the background fine-scale FESpace of order q
  reffe = ReferenceFE(lagrangian, Float64, q)
  Uh = TestFESpace(Ωf, reffe, conformity=:H1)
  # Build the full matrices
  K = assemble_stima(Uh, A, qorder);
  L = assemble_rect_matrix(Ωc, Uh, p, elem_global_unique_node_ids);
  Λ = assemble_rhs_matrix(Ωc, p)

  interior_global_dofs = Ωms.interior_boundary_global[1] # Since we have zero boundary conditions, we extract only the interior nodes

  num_coarse_cells = num_cells(Ωc)
  n_monomials = (p+1)^2
  elem_to_dof(x) = n_monomials*x-n_monomials+1:n_monomials*x;
  coarse_dofs = lazy_map(elem_to_dof, 1:num_coarse_cells)

  # Obtain the basis functions
  basis_vec_ms = lazy_map(get_ms_bases, 
    lazy_fill(K, num_coarse_cells), 
    lazy_fill(L, num_coarse_cells), 
    lazy_fill(Λ, num_coarse_cells), 
    interior_global_dofs, 
    coarse_dofs);

  MultiScaleFESpace(Ωms, Uh, basis_vec_ms)
end

"""
Function to obtain the multiscale bases functions
"""

function get_ms_bases(stima::SparseMatrixCSC{Float64, Int64}, lmat::SparseMatrixCSC{Float64,Int64}, rhsmat::SparseMatrixCSC{Float64,Int64}, interior_dofs, coarse_dofs)
  n_fine_dofs = size(lmat, 1)
  n_coarse_dofs = length(coarse_dofs)
  basis_vec_ms = spzeros(Float64, n_fine_dofs, n_coarse_dofs)    
  I = interior_dofs
  J = coarse_dofs
  patch_stima = stima[I, I]
  patch_lmat = lmat[I, J]
  patch_rhs = rhsmat[J, J]
  LHS = saddle_point_system(patch_stima, patch_lmat)  
  RHS = [zeros(Float64, size(I,1), size(J,1)); collect(patch_rhs)]      
  luA = lu(LHS)  
  ldiv!(luA, RHS)  
  for j=1:lastindex(J), i=1:lastindex(I)        
    basis_vec_ms[I[i],j] = RHS[i,j]        
  end    
  basis_vec_ms
end

function get_boundary_indices_direct(σ)
  c = groupcount(combinedims(σ))
  [k for (k,v) in zip(c.indices,c.values) if (v==1) || (v==2) || (v==3)]
end

function get_interior_indices_direct(σ)
  c = groupcount(combinedims(σ))
  [k for (k,v) in zip(c.indices,c.values) if (v==4)]
end