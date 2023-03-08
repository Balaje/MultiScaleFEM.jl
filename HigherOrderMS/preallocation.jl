##### ###### ###### ###### ###### ###### ###### ###### ###### ###### #####
# Preallocate important data structures and empty arrays for perfomance
##### ###### ###### ###### ###### ###### ###### ###### ###### ###### #####


function preallocate_matrices(domain::Tuple{Float64,Float64}, nc::Int64, nf::Int64, l::Int64, fespaces::Tuple{Int64,Int64},
  f::Function, D::Function, φᵢ!::Function, ∇φᵢ!::Function)
  # Define the aspect ratio between the coarse and fine scale
  aspect_ratio = Int(nf/nc)  
  @assert abs(Int(nf/nc) - (nf/nc)) == 0.0 # Determine if the aspect ratio is an integer.
  
  q,p = fespaces
  nds_coarse = LinRange(domain[1], domain[2], nc+1)
  nds_fine = LinRange(domain[1], domain[2], nf+1)
  elem_coarse = [i+j for i=1:nc, j=0:1]
  elem_fine = [i+j for i=1:nf, j=0:1]
  # To be defined
  patch_elem_indices_to_fine_elem_indices = Vector{AbstractVector{Int64}}(undef, nc) # Mapping between the patch elements to the fine scale 
  coarse_elem_indices_to_fine_elem_indices = Vector{AbstractVector{Int64}}(undef, nc) # Mapping between the coarse elements to the fine scale     
  # Empty matrices
  multiscale_elem = Vector{Vector{Int64}}(undef,nc) # Connectivity of the multiscale elements
  basis_vec_multiscale = Vector{Matrix{Float64}}(undef,nc) # To store the multiscale basis functions
  basis_elem_multiscale = Vector{Matrix{Float64}}(undef,nc)
  # Preallocate the shape of a multiscale matrix-vector system
  multiscale_matrix = zeros(Float64, (p+1)*nc, (p+1)*nc)
  multiscale_vector = zeros(Float64, (p+1)*nc)  
  ip_elem_cache = Vector{Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Float64}}}(undef,nc)
  for i=1:nc
    start = max(1,i-l)
    last = min(nc,i+l)
    npatch = (last-start+1)    
    patch_elem_indices_to_fine_elem_indices[i] = (1:(q*aspect_ratio)*npatch+1) .+ ((start-1)*(q*aspect_ratio))
    coarse_elem_indices_to_fine_elem_indices[i] = i*(q*aspect_ratio) - (q*aspect_ratio) + 1 : i*(q*aspect_ratio) + 1
    multiscale_elem[i] = start*(p+1)-p : last*(p+1)    
    basis_vec_multiscale[i] = zeros(Float64, q*nf+1, p+1)    
    basis_elem_multiscale[i] = zeros(Float64, q*nf+1, npatch*(p+1))    
    ip_elem_cache[i] = zeros(Float64, npatch*(p+1), npatch*(p+1)), zeros(Float64, length(coarse_elem_indices_to_fine_elem_indices[i]), npatch*(p+1)), zeros(Float64, npatch*(p+1), length(coarse_elem_indices_to_fine_elem_indices[i])), zeros(Float64, npatch*(p+1))
  end
  L = zeros(Float64, length(coarse_elem_indices_to_fine_elem_indices[1]))
  Lt = zero(L)
  ipcache = zero(L) # Cache for efficiently computing the inner product
  # Store the data cell-wise
  Ds = convert_to_cell_wise(D, nc)
  fs = convert_to_cell_wise(f, nc)
  Φs = convert_to_cell_wise(φᵢ!, nc)
  ∇Φs = convert_to_cell_wise(∇φᵢ!, nc)
  jacobian_exp = convert_to_cell_wise(1, nc)
  nc_elem = convert_to_cell_wise(nc, nc)
  p_elem = convert_to_cell_wise(p, nc)
  l_elem = convert_to_cell_wise(l, nc)
  (nds_coarse, elem_coarse, nds_fine, elem_fine), patch_elem_indices_to_fine_elem_indices, coarse_elem_indices_to_fine_elem_indices,
  (basis_vec_multiscale, basis_elem_multiscale), (multiscale_elem, multiscale_matrix, multiscale_vector), 
  (Ds, fs, Φs, ∇Φs, jacobian_exp, nc_elem, p_elem, l_elem, ip_elem_cache) 
end

get_node_elem_coarse(prob_data) = (prob_data[1][1], prob_data[1][2])
get_node_elem_fine(prob_data) = (prob_data[1][3], prob_data[1][4])
get_patch_indices_to_global_indices(prob_data) = prob_data[2]
get_coarse_indices_to_fine_indices(prob_data) = prob_data[3]
get_basis_multiscale(prob_data) = prob_data[4]
get_multiscale_data(prob_data) = prob_data[5]
get_elem_wise_data(prob_data) = prob_data[6]