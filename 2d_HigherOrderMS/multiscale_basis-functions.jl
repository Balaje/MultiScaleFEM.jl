# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
# File containing the code to extract the 2d patch and compute the multiscale basis functions  #
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

struct FineScaleSpace <: FESpace
  U::FESpace
  elemTree::BruteTree
end
function FineScaleSpace(domain::Tuple{Float64,Float64,Float64,Float64}, q::Int64, nf::Int64)
  model = simplexify(CartesianDiscreteModel(domain, (nf,nf)))
  reffe = ReferenceFE(lagrangian, Float64, q)
  U = TestFESpace(model, reffe, conformity=:H1)
  Ω = get_triangulation(U)
  σ = get_cell_node_ids(Ω)
  R = vec(map(x->SVector(Tuple(x)), σ))
  tree = BruteTree(R, ElemDist())
  FineScaleSpace(U, tree)
end

struct ElemDist <: NearestNeighbors.Distances.Metric end
function NearestNeighbors.Distances.evaluate(::ElemDist, x::AbstractVector, y::AbstractVector)
  dist = abs(x[1] - y[1])
  for i=1:lastindex(x), j=1:lastindex(y)
    dist = min(dist, abs(x[i]-y[j]))
  end
  dist+1
end

function get_patch_elem_inds(fine_scale_space::FineScaleSpace, l::Int64, el::Int64)
  U = fine_scale_space.U
  tree = fine_scale_space.elemTree
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
function get_patch_global_node_ids(fine_scale_space::FineScaleSpace, l::Int64, el::Int64)
  R = get_patch_elem_inds(fine_scale_space, l, el)
  U = fine_scale_space.U
  Ω = get_triangulation(U)
  σ = get_cell_node_ids(Ω)
  lazy_map(Broadcasting(Reindex(σ)), R)
end

# Local indices for solving the patch problems
function get_patch_local_to_global_node_inds(fine_scale_space::FineScaleSpace, l::Int64, el::Int64)
  Q = get_patch_global_node_ids(fine_scale_space, l, el)
  sort(unique(mapreduce(permutedims, vcat, Q)))
end

function get_patch_local_node_ids(fine_scale_space::FineScaleSpace, l::Int64, el::Int64)
  σ = collect(get_patch_global_node_ids(fine_scale_space, l, el))
  R = get_patch_local_to_global_node_inds(fine_scale_space, l, el)
  for t=1:lastindex(σ)
    for tt = 1:lastindex(R), ttt = 1:lastindex(σ[t])
      if(σ[t][ttt] == R[tt])
        σ[t][ttt] = tt
      end
    end
  end
  σ
end

function get_patch_cell_types(fine_scale_space::FineScaleSpace, l::Int64, el::Int64)
  U = fine_scale_space.U
  cell_type = get_cell_type(get_triangulation(U))
  R = get_patch_elem_inds(fine_scale_space, l, el)
  lazy_map(Broadcasting(Reindex(cell_type)), R)
end
function get_patch_node_coordinates(fine_scale_space, l::Int64, el::Int64)
  U = fine_scale_space.U
  Ω = get_triangulation(U)
  C = get_node_coordinates(Ω)
  R = get_patch_local_to_global_node_inds(fine_scale_space, l, el)
  C[R]
end