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

function get_patch_cell_coordinates(node_coordinates, patch_fine_node_ids)
  b = Broadcasting(Reindex(node_coordinates))
  lazy_map(Broadcasting(b), patch_fine_node_ids)
end

function get_patch_local_cell_ids(patch_fine_elems, σ)
  patch_fine_node_ids = collect(lazy_map(Broadcasting(Reindex(σ)), patch_fine_elems))
  R = sort(unique(mapreduce(permutedims, vcat, patch_fine_node_ids)))
  for t = 1:lastindex(patch_find_node_ids)
    for tt = 1:lastindex(R), ttt = 1:lastindex(patch_fine_node_ids[t])
      if(patch_fine_node_ids[t][ttt] == R[tt])
        patch_fine_node_ids[t][ttt] = tt  
      end
    end 
  end
  patch_fine_node_ids
end

function get_patch_node_coordinates(cell_coordinates)
  unique(reduce(vcat, cell_coordinates))
end

function get_patch_cell_type(cell_types, patch_elem_indices)
  lazy_map(Broadcasting(Reindex(cell_types)), patch_elem_indices)
end