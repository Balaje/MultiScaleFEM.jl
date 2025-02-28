##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ##### ##### ##### 
# Compute the mapping between the coarse-scale and the fine-scale discretizations ####
##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ##### ##### #####

function coarse_space_to_fine_space(nc::Int64, nf::Int64, l::Int64, fespaces::Tuple{Int64,Int64})
  # Define the aspect ratio between the coarse and fine scale
  aspect_ratio = Int(nf/nc)  
  @assert abs(Int(nf/nc) - (nf/nc)) == 0.0 # Determine if the aspect ratio is an integer.  
  q,p = fespaces
  
  patch_elem_indices_to_fine_elem_indices = Vector{AbstractVector{Int64}}(undef, nc) # Mapping between the patch elements to the fine scale 
  coarse_elem_indices_to_fine_elem_indices = Vector{AbstractVector{Int64}}(undef, nc) # Mapping between the coarse elements to the fine scale     
  multiscale_elem = Vector{Vector{Int64}}(undef,nc) # Connectivity of the multiscale elements

  for i=1:nc
    start = max(1,i-l)
    last = min(nc,i+l)
    npatch = (last-start+1)    
    patch_elem_indices_to_fine_elem_indices[i] = (1:(q*aspect_ratio)*npatch+1) .+ ((start-1)*(q*aspect_ratio))
    coarse_elem_indices_to_fine_elem_indices[i] = i*(q*aspect_ratio) - (q*aspect_ratio) + 1 : i*(q*aspect_ratio) + 1
    multiscale_elem[i] = start*(p+1)-p : last*(p+1)    
  end
  
  patch_elem_indices_to_fine_elem_indices, coarse_elem_indices_to_fine_elem_indices, multiscale_elem
end