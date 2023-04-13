##### # ##### # ##### # ##### # ##### # ##### # ##### # ##### # ##### # ##### # ##### # ##### #
# Functions to compute the coarse-to-fine map
# Contains the element indices in the fine scale corresponding to the coarse scale element
# Currently this works only for simplices. Has to be generalized for any meshes
##### # ##### # ##### # ##### # ##### # ##### # ##### # ##### # ##### # ##### # ##### # ##### #

function _source_to_target(A, B)
  b = Broadcasting(Reindex(B))
  r = lazy_map(b, A)
end
function _refine_once(j::Int64)    
  # This implementation exploits the pattern in the refinement of uniform simplicial mesh.
  # Needs to be generalized for arbitrary meshes.
  check_elem(k, j) = (k%2^(j+1) == 1)
  num_cells = 2^(2*j+1)
  ref = Vector{Vector{Int64}}(undef, num_cells)
  ref[1] = [1,2,3,2^(j+2)+1]
  for k=2:num_cells
    prev = sort(ref[k-1], rev=true)
    lift = check_elem(k,j)*2^(j+2) # Lift up one row if the edge elements are encountered
    if((k % 2) == 0)
      ref[k] = [prev[2]+1; prev[1]+1:prev[1]+3] .+ lift # Even elements
    else
      ref[k] = [prev[end]+1:prev[end]+3; prev[1]+1] .+ lift # Odd elements
    end
  end
  ref
end
function get_coarse_to_fine_map(num_coarse_cells::Int64, num_fine_cells::Int64)        
  j_coarse = ceil(Int64, 0.5*(log2(num_coarse_cells)-1))    
  j_fine = ceil(Int64, 0.5*(log2(num_fine_cells)-1))
  all_j = j_coarse+1:j_fine
  c_to_f_maps = [_refine_once(j-1) for j in all_j]
  (length(c_to_f_maps) == 1) && return c_to_f_maps[1]
  c_to_f = _source_to_target(c_to_f_maps[end-1], c_to_f_maps[end])
  for l=lastindex(all_j)-1:-1:2
    c_to_f = _source_to_target(c_to_f_maps[l-1], c_to_f)
  end
  c_to_f
end 