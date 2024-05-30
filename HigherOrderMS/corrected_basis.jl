##### ###### ###### ###### ###### ###### ###### ###### #
# Program to implement the corrected basis function
##### ###### ###### ###### ###### ###### ###### ###### #


function compute_corrected_basis_function(fine_scale_space::FineScaleSpace, D::Function, p::Int64, nc::Int64, l::Int64, patch_indices_to_global_indices::Vector{AbstractVector{Int64}}, σ₀, σ₁)
  ### To build the basis functions
  nf = fine_scale_space.nf
  q = fine_scale_space.q
  basis_vec_ms = spzeros(Float64,q*nf+1,(p+1)*nc) # To store the new multiscale basis functions

  K, L, Λ = get_saddle_point_problem(fine_scale_space, D, p, nc)
  M = assemble_mass_matrix(fine_scale_space, x->1.0)

  f1 = zeros(Float64,size(K,1))
  index = 1
  for t=1:nc  
    fullnodes = patch_indices_to_global_indices[t]
    bnodes = [fullnodes[1], fullnodes[end]]
    bvals = [0.0,0.0]
    freenodes = setdiff(fullnodes, bnodes)
    start = max(1,t-l)
    last = min(nc,t+l)
    gn = start*(p+1)-p:last*(p+1)    
    stima_el = K[freenodes,freenodes]
    massma_el = M[freenodes,freenodes]
    lmat_el = L[freenodes,gn]
    loadvec_el = (f1 - K[:,bnodes]*bvals)
    for _=1:p+1
      fvecs_el = [loadvec_el[freenodes]; Λ[gn, index]]
      lhs = [σ₀*stima_el + σ₁*massma_el lmat_el; (lmat_el)' spzeros(Float64, length(gn), length(gn))]
      rhs = fvecs_el           
      sol = lhs\collect(rhs)
      basis_vec_ms[freenodes,index] = sol[1:length(freenodes)]
      index+=1
    end
  end
  basis_vec_ms
end
