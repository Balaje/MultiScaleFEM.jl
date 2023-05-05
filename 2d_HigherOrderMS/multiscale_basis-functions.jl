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
  collect_all_elems(X) = reduce(vcat, X)
  map(collect_all_elems, X)
end

"""
Function to obtain the global node (in the fine scale) indices of each patch.
Takes the indices of the fine scale elements in the patch and applies the connectivity matrix of the fine scale elements.
"""
get_patch_global_node_ids(patch_fine_elems, σ_fine) = lazy_map(Broadcasting(Reindex(σ_fine)), patch_fine_elems)

"""
Function to obtain the coarse-element wise dictionary containing the mapping between the global (fine-scale) and local indices.
Used to convert (global => local).
This map is essentially used to construct the patch-wise fine-scale DiscreteModel.
"""
function get_patch_wise_global_to_local_map(patch_fine_elems, σ)
  patch_fine_node_ids = get_patch_global_node_ids(patch_fine_elems, σ)
  unique_patch_fine_node_ids = map(get_unique_node_ids, patch_fine_node_ids)  
  map(get_local_node_ids, unique_patch_fine_node_ids)
end

"""
Function to obtain the coarse-element wise Dict containing the mapping between the global (fine-scale) and local indices.
Used to convert (local => global).
This map is used to take back local indices to global indices in order to extract the entries from the global (fine-scale) stiffness matrix
"""
function get_patch_wise_local_to_global_map(patch_fine_elems, σ)
  patch_fine_node_ids = get_patch_global_node_ids(patch_fine_elems, σ)
  unique_patch_fine_node_ids = map(get_unique_node_ids, patch_fine_node_ids)
  map(get_global_node_ids, unique_patch_fine_node_ids) 
end

get_local_node_ids(X) = Dict(zip(X, 1:length(X))); # Function to obtain the pair describing the global-local ids
get_global_node_ids(X) = Dict(zip(1:length(X), X)); # Function to obtain the pair describing the local-global ids

"""
Function to assign the global node (in the fine scale) indices a local index for the Gridap Mesh Generator.
This will then be used to generate the patch Triangulation and then extract the interior and boundary degrees of freedom in the local sense.
"""
function get_patch_local_node_ids(patch_fine_elems, σ)
  patch_fine_node_ids = get_patch_global_node_ids(patch_fine_elems, σ)
  global_to_local = get_patch_wise_global_to_local_map(patch_fine_elems, σ)
  X = map(convert_node_ids, patch_fine_node_ids, global_to_local)
  map(Table, X)
end

"""
Function to perform the transformation between the indices
"""
function convert_node_ids(v, d)   
  w = zero.(v)
  for i in 1:lastindex(v)
    for j in 1:lastindex(v[i])
      w[i][j] = get(d, v[i][j], v[i][j])
    end
  end
  w
end
function convert_dofs(v::Vector{T}, d) where T <: Integer
  w = zero(v)
  for i in 1:lastindex(v)
    w[i] = get(d, v[i], v[i])
  end
  w
end

"""
Extract the boundary nodal indices from the DiscreteModel
"""
function get_local_indices(model::DiscreteModel, str::String)
  fl = get_face_labeling(model)
  findnz(sparse(get_face_tag(fl, str, 0)))[1]
end

"""
Function to get the unique node identifiers from the connectivity matrices.
"""
get_unique_node_ids(X) = sort(unique(mapreduce(permutedims, vcat, X))); # Function to obtain the unique node ids

"""
Function to obtain the node-coordinates using the unique identifiers. Required for patch-mesh generation using Gridap.
"""
function get_patch_node_coordinates(node_coordinates, patch_fine_node_ids)
  R = get_unique_node_ids(patch_fine_node_ids)
  node_coordinates[R]
end

"""
Function to return the cell-types of the fine scale elements. Required for patch-mesh generation using Gridap.
"""
function get_patch_cell_type(cell_types, patch_elem_indices)
  lazy_map(Broadcasting(Reindex(cell_types)), patch_elem_indices) |> collect
end

function _patch_model_data(σ, nodes, cell_types, elems, num_coarse_cells)
  ids = lazy_map(Broadcasting(Reindex(σ)), elems) |> collect
  local_ids = get_patch_local_node_ids(elems, σ) |> collect
  node_coords = lazy_map(get_patch_node_coordinates, lazy_fill(nodes, num_coarse_cells), ids) |> collect
  cell_types = get_patch_cell_type(cell_types, elems) |> collect
  node_coords, local_ids, cell_types
end

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
  patch_models_coarse::Vector{DiscreteModel}
  patch_models_fine::Vector{DiscreteModel}
  interior_boundary_local
  interior_boundary_global
  local_to_global_map
end
function MultiScaleTriangulation(domain::Tuple, nf::Int64, nc::Int64, l::Int64)
  # Fine Scale Space
  model_h = simplexify(CartesianDiscreteModel(domain, (nf,nf)))
  Ωf = Triangulation(model_h)
  # Coarse Scale Space
  model_H = simplexify(CartesianDiscreteModel(domain, (nc,nc)))
  Ωc = Triangulation(model_H)
  # Store the tree of the coarse mesh for obtaining the patch
  σ = get_cell_node_ids(Ωc)
  R = vec(map(x->SVector(Tuple(x)), σ))
  tree = BruteTree(R, ElemDist())
  # Get the data structures required to build the discrete DiscreteModels
  num_coarse_cells = num_cells(Ωc)
  num_fine_cells = num_cells(Ωf)
  # 1) Get the element indices inside the patch on the coarse and fine scales
  coarse_to_fine_elems = get_coarse_to_fine_map(num_coarse_cells, num_fine_cells)
  patch_coarse_elems = map(get_patch_coarse_elem, 
    lazy_fill(Ωc, num_coarse_cells), 
    lazy_fill(tree, num_coarse_cells), 
    lazy_fill(l, num_coarse_cells), 
    1:num_coarse_cells)
  patch_fine_elems = get_patch_fine_elems(patch_coarse_elems, coarse_to_fine_elems)
  # # 2) Get the maps required to build the discrete models
  σ_coarse = get_cell_node_ids(Ωc)
  node_coordinates_coarse = get_node_coordinates(Ωc)
  cell_types_coarse = get_cell_type(Ωc)
  σ_fine = get_cell_node_ids(Ωf)
  node_coordinates_fine = get_node_coordinates(Ωf)
  cell_types_fine = get_cell_type(Ωf)

  # Get the local-global and global-local map
  local_to_global_map = get_patch_wise_local_to_global_map(patch_fine_elems, σ_fine)  

  # Obtain the patch models in coarse scale
  patch_coarse_coords, patch_coarse_local_ids, patch_coarse_cell_types = _patch_model_data(σ_coarse, node_coordinates_coarse, cell_types_coarse, patch_coarse_elems, num_coarse_cells)
  patch_grids = map(UnstructuredGrid, patch_coarse_coords, patch_coarse_local_ids, lazy_fill(get_reffes(Ωc), num_coarse_cells), patch_coarse_cell_types)
  grids_topology = map(GridTopology, patch_grids)
  face_labelling = map(FaceLabeling, grids_topology)
  patch_models_coarse = map(DiscreteModel, patch_grids, grids_topology, face_labelling)

  # Obtain the patch models in fine scale
  patch_fine_coords, patch_fine_local_ids, patch_fine_cell_types = _patch_model_data(σ_fine, node_coordinates_fine, cell_types_fine, patch_fine_elems, num_coarse_cells)
  patch_grids = map(UnstructuredGrid, patch_fine_coords, patch_fine_local_ids, lazy_fill(get_reffes(Ωf), num_coarse_cells), patch_fine_cell_types)
  grids_topology = map(GridTopology, patch_grids)
  face_labelling = map(FaceLabeling, grids_topology)
  patch_models_fine = map(DiscreteModel, patch_grids, grids_topology, face_labelling)

  # Get the boundary and interior dofs
  patch_local_interior_dofs = map(get_local_indices, patch_models_fine, lazy_fill("interior", num_coarse_cells));
  patch_local_boundary_dofs = map(get_local_indices, patch_models_fine, lazy_fill("boundary", num_coarse_cells));
  patch_global_interior_dofs = map(convert_dofs, patch_local_interior_dofs, local_to_global_map)
  patch_global_boundary_dofs = map(convert_dofs, patch_local_boundary_dofs, local_to_global_map)
  interior_boundary_global = (patch_global_interior_dofs, patch_global_boundary_dofs)
  interior_boundary_local = (patch_local_interior_dofs, patch_local_boundary_dofs)

  # Lastly, obtain the global-local map on each elements
  elem_fine = map(Broadcasting(Reindex(coarse_to_fine_elems)), 1:num_coarse_cells)
  elem_global_node_ids = map(Broadcasting(Reindex(σ_fine)), elem_fine)

  # Return the Object
  MultiScaleTriangulation(Ωc, 
  Ωf, 
  patch_models_coarse, 
  patch_models_fine, 
  interior_boundary_local, 
  interior_boundary_global, 
  elem_global_node_ids)
end


struct MultiScaleFESpace <: FESpace
  Ω::MultiScaleTriangulation
  Uh::FESpace
  basis_vec_ms::SparseMatrixCSC{Float64,Int64}
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
  elem_global_node_ids = Ωms.local_to_global_map
  elem_global_unique_node_ids = map(get_unique_node_ids, elem_global_node_ids)
  # Build the background fine-scale FESpace of order q
  reffe = ReferenceFE(lagrangian, Float64, q)
  Uh = TestFESpace(Ωf, reffe, conformity=:H1)
  # Build the full matrices
  K = assemble_stima(Uh, A, qorder);
  L = assemble_rect_matrix(Ωc, Uh, p, elem_global_unique_node_ids);
  Λ = assemble_rhs_matrix(Ωc, p)

  interior_global, _ = Ωms.interior_boundary_global # Since we have zero boundary conditions

  # Obtain the basis functions
  basis_vec_ms = get_ms_bases(K, L, Λ, interior_global, p); 

  MultiScaleFESpace(Ωms, Uh, basis_vec_ms)
end

"""
Function to obtain the multiscale bases functions
"""
function get_ms_bases(stima::SparseMatrixCSC{Float64, Int64}, 
  lmat::SparseMatrixCSC{Float64,Int64}, 
  rhsmat::SparseMatrixCSC{Float64,Int64}, 
  interior_dofs, 
  p::Int64)    
  n_fine_dofs = size(lmat, 1)
  n_coarse_dofs = size(lmat, 2)
  num_coarse_cells = size(interior_dofs, 1)
  coarse_dofs = [3p*i-3p+1:3p*i for i in 1:num_coarse_cells]
  basis_vec_ms = spzeros(Float64, n_fine_dofs, n_coarse_dofs)
  patch_stima = map(getindex, lazy_fill(stima, num_coarse_cells), interior_dofs, interior_dofs);
  patch_lmat = map(getindex, lazy_fill(lmat, num_coarse_cells), interior_dofs, coarse_dofs);
  patch_rhs = map(getindex, lazy_fill(rhsmat, num_coarse_cells), coarse_dofs, coarse_dofs);
  for i=1:num_coarse_cells
    LHS = saddle_point_system(patch_stima[i], patch_lmat[i])
    RHS = [zeros(Float64, size(interior_dofs[i],1), size(coarse_dofs[i],1)); collect(patch_rhs[i])]
    sol = LHS\RHS
    for j=1:lastindex(coarse_dofs[i])
      basis_vec_ms[interior_dofs[i], coarse_dofs[i][j]] = sol[1:length(interior_dofs[i]), j]
    end
  end
  basis_vec_ms
end