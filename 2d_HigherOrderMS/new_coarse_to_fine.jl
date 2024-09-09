##### ##### ##### ##### ##### ##### ##### ##### 
# Script to define the refinement strategy
##### ##### ##### ##### ##### ##### ##### ##### 
using BlockArrays
using SplitApplyCombine

function coarsen(model::DiscreteModel, ntime::Int64)
  nc = num_cells(model)
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

