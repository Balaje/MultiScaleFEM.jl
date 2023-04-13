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

function get_patch_elem_inds(ms_space::MultiScaleFESpace, l::Int64, el::Int64)
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
function get_patch_global_node_ids(ms_space::MultiScaleFESpace, l::Int64, el::Int64)
  R = get_patch_elem_inds(ms_space, l, el)
  U = ms_space.UH
  Ω = get_triangulation(U)
  σ = get_cell_node_ids(Ω)
  lazy_map(Broadcasting(Reindex(σ)), R)
end

# Local indices for solving the patch problems
function get_patch_local_to_global_node_inds(ms_space::MultiScaleFESpace, l::Int64, el::Int64)
  Q = get_patch_global_node_ids(ms_space, l, el)
  sort(unique(mapreduce(permutedims, vcat, Q)))
end

function get_patch_local_node_ids(ms_space::MultiScaleFESpace, l::Int64, el::Int64)
  σ = collect(get_patch_global_node_ids(ms_space, l, el))
  R = get_patch_local_to_global_node_inds(ms_space, l, el)
  for t=1:lastindex(σ)
    for tt = 1:lastindex(R), ttt = 1:lastindex(σ[t])
      if(σ[t][ttt] == R[tt])
        σ[t][ttt] = tt
      end
    end
  end
  σ
end

function get_patch_cell_types(ms_space::MultiScaleFESpace, l::Int64, el::Int64)
  U = ms_space.UH
  cell_type = get_cell_type(get_triangulation(U))
  R = get_patch_elem_inds(ms_space, l, el)
  lazy_map(Broadcasting(Reindex(cell_type)), R)
end
function get_patch_node_coordinates(ms_space::MultiScaleFESpace, l::Int64, el::Int64)
  U = ms_space.UH
  Ω = get_triangulation(U)
  C = get_node_coordinates(Ω)
  R = get_patch_local_to_global_node_inds(ms_space, l, el)
  C[R]
end