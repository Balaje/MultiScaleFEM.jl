# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
# File containing the code to extract the 2d patch and compute the multiscale basis functions  #
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

struct MultiScaleFESpace <: FESpace
  UH::FESpace
  Uh::FESpace
  elemTree::BruteTree
  patch_coords::Tuple{Any,Any}
  patch_global_ids::Tuple{Any,Any}
  patch_local_ids::Tuple{Any,Any}
  patch_cell_types::Tuple{Any,Any}
  coarse_to_fine::Tuple{Any,Any,Any,Any}
end
function MultiScaleFESpace(domain::Tuple{Float64,Float64,Float64,Float64}, q::Int64, p::Int64, nf::Int64, nc::Int64, l::Int64)
  # Fine Scale Space
  model_h = simplexify(CartesianDiscreteModel(domain, (nf,nf)))
  reffe_h = ReferenceFE(lagrangian, Float64, q)
  Uh = TestFESpace(model_h, reffe_h, conformity=:H1)
  # Coarse Scale Space
  model_H = simplexify(CartesianDiscreteModel(domain, (nc,nc)))
  reffe_H = ReferenceFE(lagrangian, Float64, p)
  UH = TestFESpace(model_H, reffe_H, conformity=:L2)
  # Store the tree of the coarse mesh for obtaining the patch
  Ω = get_triangulation(UH)
  σ = get_cell_node_ids(Ω)
  R = vec(map(x->SVector(Tuple(x)), σ))
  tree = BruteTree(R, ElemDist())
  
  ####
  # Get the data structures required to build the discrete DiscreteModels
  ####
  num_coarse_cells = num_cells(get_triangulation(UH))
  num_fine_cells = num_cells(get_triangulation(Uh))

  # 1) Get the element indices inside the patch on the coarse and fine scales
  coarse_to_fine_elems = get_coarse_to_fine_map(num_coarse_cells, num_fine_cells)
  patch_coarse_elems = lazy_map(get_patch_coarse_elem, Gridap.Arrays.Fill(UH, num_coarse_cells),
      Gridap.Arrays.Fill(tree, num_coarse_cells), Gridap.Arrays.Fill(l, num_coarse_cells), 1:num_coarse_cells)
  patch_fine_elems = get_patch_fine_elems(patch_coarse_elems, coarse_to_fine_elems)

  # 2) Get the maps required to build the discrete models
  σ_coarse = get_cell_node_ids(get_triangulation(UH))
  node_coordinates_coarse = get_node_coordinates(get_triangulation(UH))
  cell_types_coarse = get_cell_type(get_triangulation(UH))
  σ_fine = get_cell_node_ids(get_triangulation(Uh))
  node_coordinates_fine = get_node_coordinates(get_triangulation(Uh))
  cell_types_fine = get_cell_type(get_triangulation(Uh))

  patch_coarse_coords, patch_coarse_global_ids, patch_coarse_local_ids, patch_coarse_cell_types = _patch_model_data(σ_coarse, node_coordinates_coarse, cell_types_coarse, patch_coarse_elems, num_coarse_cells)
  patch_fine_coords, patch_fine_global_ids, patch_fine_local_ids, patch_fine_cell_types = _patch_model_data(σ_fine, node_coordinates_fine, cell_types_fine, patch_fine_elems, num_coarse_cells)

  # Las
  elems = get_patch_fine_elems([i for i in 1:num_coarse_cells], coarse_to_fine_elems)
  coarse_to_fine = _patch_model_data(σ_fine, node_coordinates_fine, cell_types_fine, elems, num_coarse_cells)

  patch_coords = (patch_coarse_coords, patch_fine_coords)
  patch_global_ids = (patch_coarse_global_ids, patch_fine_global_ids)
  patch_local_ids = (patch_coarse_local_ids, patch_fine_local_ids)
  patch_cell_types = (patch_coarse_cell_types, patch_fine_cell_types)

  # Return the Object
  MultiScaleFESpace(UH, Uh, tree, patch_coords, patch_global_ids, patch_local_ids, patch_cell_types, coarse_to_fine)
end

get_coarse_data(ms_space::MultiScaleFESpace) = (ms_space.patch_coords[1], ms_space.patch_global_ids[1], ms_space.patch_local_ids[1], ms_space.patch_cell_types[1])
get_fine_data(ms_space::MultiScaleFESpace) = (ms_space.patch_coords[2], ms_space.patch_global_ids[2], ms_space.patch_local_ids[2], ms_space.patch_cell_types[2])
get_elem_data(ms_space::MultiScaleFESpace) = ms_space.coarse_to_fine

struct ElemDist <: NearestNeighbors.Distances.Metric end
function NearestNeighbors.Distances.evaluate(::ElemDist, x::AbstractVector, y::AbstractVector)
  dist = abs(x[1] - y[1])
  for i=1:lastindex(x), j=1:lastindex(y)
    dist = min(dist, abs(x[i]-y[j]))
  end
  dist+1
end

# function get_patch_coarse_elem(ms_space::MultiScaleFESpace, l::Int64, el::Int64)
function get_patch_coarse_elem(coarse_space::FESpace, tree::BruteTree, l::Int64, el::Int64)  
  Ω = get_triangulation(coarse_space)
  σ = get_cell_node_ids(Ω)
  el_inds = inrange(tree, σ[el], 1) # Find patch of size 1
  for _=2:l # Recursively do this for 2:l and collect the unique indices. 
    X = [inrange(tree, i, 1) for i in σ[el_inds]]
    el_inds = unique(vcat(X...))
  end
  sort(el_inds)
  # There may be a better way to do this... Need to check.
end

function get_patch_fine_elems(patch_coarse_elems, coarse_to_fine_elem)
  Y = reduce.(vcat, lazy_map(Broadcasting(Reindex(coarse_to_fine_elem)), patch_coarse_elems))
  sort.(Y)
end

function get_patch_global_node_ids(patch_fine_elems, σ)
  collect(lazy_map(Broadcasting(Reindex(σ)), patch_fine_elems))
end

function get_patch_local_node_ids(patch_fine_elems, σ)
  patch_fine_node_ids = get_patch_global_node_ids(patch_fine_elems, σ)
  unique_patch_fine_node_ids = lazy_map(get_unique_node_ids, patch_fine_node_ids)
  global_to_local = lazy_map(get_local_node_ids, unique_patch_fine_node_ids)
  lazy_map(convert_global_to_local_node_ids, global_to_local, patch_fine_node_ids)
end

# Function to obtain the unique node ids
get_unique_node_ids(X) = sort(unique(mapreduce(permutedims, vcat, X)))
# Build the dictionary that maps the global node indices to local
get_local_node_ids(X) = Dict(X[i] => i for i in 1:length(X))
convert_global_to_local_node_ids(d, global_node_ids) = lazy_map(Broadcasting(global_node_ids), d)

function get_patch_node_coordinates(node_coordinates, patch_fine_node_ids)
  R = sort(unique(mapreduce(permutedims, vcat, patch_fine_node_ids)))
  node_coordinates[R]
end

function get_patch_cell_type(cell_types, patch_elem_indices)
  lazy_map(Broadcasting(Reindex(cell_types)), patch_elem_indices)
end

struct FineScale end
function get_patch_triangulation(ms_space::MultiScaleFESpace, num_coarse_cells::Int64, ::FineScale)
  # Get the fine-scale (patch-local) coordinates, (patch-local) cell-ids and (patch-local) cell_types
  patch_node_coordinates, _, patch_fine_local_node_ids, patch_cell_types = get_fine_data(ms_space)
  
  # Construct the grids from the node coordinates and the connectivity
  patch_grids = lazy_map(UnstructuredGrid, patch_node_coordinates, patch_fine_local_node_ids, Gridap.Arrays.Fill(get_reffes(get_triangulation(ms_space.Uh)), num_coarse_cells), patch_cell_types)
  patch_grids_topology = lazy_map(GridTopology, patch_grids)
  patch_face_labelling = lazy_map(FaceLabeling, patch_grids_topology)
  # Construct the DiscreteModels for the Triangulation
  lazy_map(DiscreteModel, patch_grids, patch_grids_topology, patch_face_labelling)
end

struct CoarseScale end
function get_patch_triangulation(ms_space::MultiScaleFESpace, num_coarse_cells::Int64, ::CoarseScale)
  # Get the coarse-scale (patch-local) coordinates, (patch-local) cell-ids and (patch-local) cell_types
  patch_node_coordinates, _, patch_coarse_local_node_ids, patch_cell_types = get_coarse_data(ms_space)
  
  patch_grids = lazy_map(UnstructuredGrid, patch_node_coordinates, patch_coarse_local_node_ids, Gridap.Arrays.Fill(get_reffes(get_triangulation(ms_space.UH)), num_coarse_cells), patch_cell_types)
  patch_grids_topology = lazy_map(GridTopology, patch_grids)
  patch_face_labelling = lazy_map(FaceLabeling, patch_grids_topology)
  # Construct the DiscreteModels for the Triangulation
  lazy_map(DiscreteModel, patch_grids, patch_grids_topology, patch_face_labelling)
end

struct FineScaleElemWise end
function get_patch_triangulation(ms_space::MultiScaleFESpace, num_coarse_cells::Int64, ::FineScaleElemWise)
  # Get the coarse-scale (patch-local) coordinates, (patch-local) cell-ids and (patch-local) cell_types
  elem_node_coordinates, _, elem_coarse_local_node_ids, elem_cell_types = get_elem_data(ms_space)
  
  patch_grids = lazy_map(UnstructuredGrid, elem_node_coordinates, elem_coarse_local_node_ids, Gridap.Arrays.Fill(get_reffes(get_triangulation(ms_space.UH)), num_coarse_cells), elem_cell_types)
  patch_grids_topology = lazy_map(GridTopology, patch_grids)
  patch_face_labelling = lazy_map(FaceLabeling, patch_grids_topology)
  # Construct the DiscreteModels for the Triangulation
  lazy_map(DiscreteModel, patch_grids, patch_grids_topology, patch_face_labelling)
end

function _patch_model_data(σ, nodes, cell_types, elems, num_coarse_cells)
  ids = collect(lazy_map(Broadcasting(Reindex(σ)), elems))
  local_ids = Table.(lazy_map(get_patch_local_node_ids, elems, Gridap.Arrays.Fill(σ,num_coarse_cells)))
  node_coords = lazy_map(get_patch_node_coordinates, Gridap.Arrays.Fill(nodes, num_coarse_cells), ids)
  cell_types = get_patch_cell_type(cell_types, elems)
  node_coords, ids, local_ids, cell_types
end