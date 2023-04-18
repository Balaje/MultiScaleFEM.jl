# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
# File containing the code to extract the 2d patch and compute the multiscale basis functions  #
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

struct MultiScaleFESpace <: FESpace
  UH::FESpace
  Uh::FESpace
  elemTree::BruteTree
end
function MultiScaleFESpace(domain::Tuple{Float64,Float64,Float64,Float64}, q::Int64, p::Int64, nf::Int64, nc::Int64)
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
  # Return the Object
  MultiScaleFESpace(UH, Uh, tree)
end

struct ElemDist <: NearestNeighbors.Distances.Metric end
function NearestNeighbors.Distances.evaluate(::ElemDist, x::AbstractVector, y::AbstractVector)
  dist = abs(x[1] - y[1])
  for i=1:lastindex(x), j=1:lastindex(y)
    dist = min(dist, abs(x[i]-y[j]))
  end
  dist+1
end

function get_patch_coarse_elem(ms_space::MultiScaleFESpace, l::Int64, el::Int64)
  U = ms_space.UH
  tree = ms_space.elemTree
  Ω = get_triangulation(U)
  σ = get_cell_node_ids(Ω)
  el_inds = inrange(tree, σ[el], 1) # Find patch of size 1
  for _=2:l # Recursively do this for 2:l and collect the unique indices. 
    X = [inrange(tree, i, 1) for i in σ[el_inds]]
    el_inds = unique(vcat(X...))
  end
  sort(el_inds)
  # There may be a better way to do this... Need to check.
end

function get_patch_fine_elems(ms_space::MultiScaleFESpace, l::Int64, num_coarse_cells::Int64, coarse_to_fine_elem)
  ms_spaces = Gridap.Arrays.Fill(ms_space, num_coarse_cells)
  ls = Gridap.Arrays.Fill(l, num_coarse_cells)
  X = lazy_map(get_patch_coarse_elem, ms_spaces, ls, 1:num_coarse_cells)
  Y = reduce.(vcat, lazy_map(Broadcasting(Reindex(coarse_to_fine_elem)), X))
  sort.(Y)
end

function get_patch_global_node_ids(patch_fine_elems, σ)
  collect(lazy_map(Broadcasting(Reindex(σ)), patch_fine_elems))
end

function get_patch_local_cell_ids(patch_fine_elems, σ)
  patch_fine_node_ids = get_patch_global_node_ids(patch_fine_elems, σ)
  R = sort(unique(mapreduce(permutedims, vcat, patch_fine_node_ids)))
  for t = 1:lastindex(patch_fine_node_ids)
    for tt = 1:lastindex(R), ttt = 1:lastindex(patch_fine_node_ids[t])
      if(patch_fine_node_ids[t][ttt] == R[tt])
        patch_fine_node_ids[t][ttt] = tt  
      end
    end 
  end
  patch_fine_node_ids
end

function get_patch_node_coordinates(node_coordinates, patch_fine_node_ids)
  R = sort(unique(mapreduce(permutedims, vcat, patch_fine_node_ids)))
  node_coordinates[R]
end

function get_patch_cell_type(cell_types, patch_elem_indices)
  lazy_map(Broadcasting(Reindex(cell_types)), patch_elem_indices)
end

function get_patch_triangulation(ms_space::MultiScaleFESpace, l::Int64, num_coarse_cells::Int64, coarse_to_fine_elems)
  node_coordinates = get_node_coordinates(get_triangulation(ms_space.Uh))
  σ = get_cell_node_ids(get_triangulation(ms_space.Uh))
  cell_types = get_cell_type(get_triangulation(ms_space.Uh))

  patch_fine_elems = get_patch_fine_elems(ms_space, l, num_coarse_cells, coarse_to_fine_elems) # Get the indices of the fine-scale elements inside the patch
  patch_fine_node_ids = get_patch_global_node_ids(patch_fine_elems, σ) # Get the global numbering of the nodes inside the patch to get the coordinates
  patch_fine_local_node_ids = Table.(lazy_map(get_patch_local_cell_ids, patch_fine_elems, Gridap.Arrays.Fill(σ,num_coarse_cells))) # Get the local numbering of the nodes inside the patch to build the triangulation
  patch_node_coordinates = lazy_map(get_patch_node_coordinates, Gridap.Arrays.Fill(node_coordinates, num_coarse_cells), patch_fine_node_ids) # Get the node coordinates in the patch
  patch_cell_types = get_patch_cell_type(cell_types, patch_fine_elems) # Get the cell types in the patch
  
  # Construct the grids from the node coordinates and the connectivity
  patch_grids = lazy_map(UnstructuredGrid, patch_node_coordinates, patch_fine_local_node_ids, Gridap.Arrays.Fill(get_reffes(get_triangulation(ms_space.Uh)), num_coarse_cells), patch_cell_types)
  patch_grids_topology = lazy_map(GridTopology, patch_grids)
  patch_face_labelling = lazy_map(FaceLabeling, patch_grids_topology)
  # Construct the DiscreteModels for the Triangulation
  lazy_map(DiscreteModel, patch_grids, patch_grids_topology, patch_face_labelling)
end

function get_patch_triangulation(ms_spaces::MultiScaleFESpace, l::Int64, num_coarse_cells::Int64)
  node_coordinates = get_node_coordinates(get_triangulation(ms_space.UH))
  σ = get_cell_node_ids(get_triangulation(ms_space.UH))
  cell_types = get_cell_type(get_triangulation(ms_space.UH))

  patch_coarse_elems = lazy_map(get_patch_coarse_elem, Gridap.Arrays.Fill(ms_spaces, num_coarse_cells), Gridap.Arrays.Fill(l, num_coarse_cells), 1:num_coarse_cells)
  patch_coarse_node_ids = collect(lazy_map(Broadcasting(Reindex(σ)), patch_coarse_elems))
  patch_coarse_local_ids = Table.(lazy_map(get_patch_local_cell_ids, patch_coarse_elems, Gridap.Arrays.Fill(σ,num_coarse_cells)))
  patch_node_coordinates = lazy_map(get_patch_node_coordinates, Gridap.Arrays.Fill(node_coordinates, num_coarse_cells), patch_coarse_node_ids)
  patch_cell_types = get_patch_cell_type(cell_types, patch_coarse_elems)

  patch_grids = lazy_map(UnstructuredGrid, patch_node_coordinates, patch_coarse_local_ids, Gridap.Arrays.Fill(get_reffes(get_triangulation(ms_space.Uh)), num_coarse_cells), patch_cell_types)
  patch_grids_topology = lazy_map(GridTopology, patch_grids)
  patch_face_labelling = lazy_map(FaceLabeling, patch_grids_topology)
  # Construct the DiscreteModels for the Triangulation
  lazy_map(DiscreteModel, patch_grids, patch_grids_topology, patch_face_labelling)
end