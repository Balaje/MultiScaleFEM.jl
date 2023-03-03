##### ###### ###### ###### ###### ###### ###### ###### ###### ###### #####
# Preallocate important data structures and empty arrays for perfomance
##### ###### ###### ###### ###### ###### ###### ###### ###### ###### #####

module PreAllocateMatrices

  function preallocate_matrices(domain::Tuple{Float64,Float64}, nc::Int64, nf::Int64, l::Int64, fespaces::Tuple{Int64,Int64})
    # Define the aspect ratio between the coarse and fine scale
    aspect_ratio = Int(nf/nc)  
    @assert abs(Int(nf/nc) - (nf/nc)) == 0.0 # Determine if the aspect ratio is an integer.

    q,p = fespaces
    nds_coarse = LinRange(domain[1], domain[2], nc+1)
    nds_fine = LinRange(domain[1], domain[2], nf+1)
    elem_coarse = [i+j for i=1:nc, j=0:1]
    elem_fine = [i+j for i=1:nf, j=0:1]
    # To be defined
    patch_elems = Vector{Int64}(undef, nc) # Enough to just store the number of elements in the patch_elems
    patch_elem_indices_to_fine_elem_indices = Vector{AbstractVector{Int64}}(undef, nc) # Mapping between the patch elements to the fine scale 
    coarse_elem_indices_to_fine_elem_indices = Vector{AbstractVector{Int64}}(undef, nc) # Mapping between the coarse elements to the fine scale     
    # Empty matrices
    multiscale_elem = Vector{Vector{Int64}}(undef,nc) # Connectivity of the multiscale elements
    basis_vec_multiscale = Vector{Matrix{Float64}}(undef,nc) # To store the multiscale basis functions

    # Preallocate the shape of a multiscale matrix 
    multiscale_matrix = Vector{Matrix{Float64}}(undef, nc)
    multiscale_vector = Vector{Vector{Float64}}(undef, nc)

    for i=1:nc
      start = max(1,i-l)
      last = min(nc,i+l)
      npatch = (last-start+1)
      patch_elems[i] = npatch
      multiscale_matrix[i] = zeros(Float64, npatch*(p+1), npatch*(p+1))
      multiscale_vector[i] = zeros(Float64, npatch*(p+1))

      patch_elem_indices_to_fine_elem_indices[i] = (1:(q*aspect_ratio)*npatch+1) .+ ((start-1)*(q*aspect_ratio))
      coarse_elem_indices_to_fine_elem_indices[i] = i*(q*aspect_ratio) - (q*aspect_ratio) + 1 : i*(q*aspect_ratio) + 1
      multiscale_elem[i] = start*(p+1)-p : last*(p+1)

      basis_vec_multiscale[i] = zeros(Float64, q*nf+1, p+1)
    end

    L = zeros(Float64, length(coarse_elem_indices_to_fine_elem_indices[1]))
    Lt = zero(L)
    ipcache = zero(L) # Cache for efficiently computing the inner product

    (nds_coarse, elem_coarse, nds_fine, elem_fine), patch_elems, patch_elem_indices_to_fine_elem_indices, coarse_elem_indices_to_fine_elem_indices, (L, Lt, ipcache),
    basis_vec_multiscale, (multiscale_elem, multiscale_matrix, multiscale_vector)

  end
end