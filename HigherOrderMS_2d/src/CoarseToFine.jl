module CoarseToFine

using Gridap
using BlockArrays
using SplitApplyCombine

function coarsen(nc::Int64, ntime::Int64)
  dims = (Int64(sqrt(nc)),Int64(sqrt(nc)))
  X = reshape(1:nc, dims) 
  m = size(X,1)
  @assert (m/2^ntime) >= 1 "Too many coarsening steps"
  block_length = Int64(m/2^ntime);  
  block_size = 2^ntime*ones(Int64,block_length)
  bX = BlockArray(Array(X), block_size, block_size)  
  collect_node_indices_inside_block.(bX.blocks)  
end

function collect_node_indices_inside_block(b::AbstractArray{T}) where T
  sort(unique(vec(b)))
end

function get_fine_nodes_in_coarse_elems(local_to_global_map, global_node_coords)  
  unique_nodes = x->unique(vec(combinedims(x)));    
  fine_node_indices = lazy_map(unique_nodes, local_to_global_map)
  fine_node_indices, lazy_map(Broadcasting(Reindex(global_node_coords)), fine_node_indices)
end

end