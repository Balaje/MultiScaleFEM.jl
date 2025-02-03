#### ##### ##### ##### ##### ##### ##### ##### 
# Module containing the functions to obtain: #           
# 1) the multiscale bases.                   #
# 2) the matrix-vector contributions.        #
#### ##### ##### ##### ##### ##### ##### #####
function compute_ms_basis(fspace::FineScaleSpace, D::Function, p::Int64, nc::Int64, l::Int64, 
  patch_indices_to_global_indices::Vector{AbstractVector{Int64}}; T=Float64)
  nf = fspace.nf
  q = fspace.q
  basis_vec_ms = spzeros(T,q*nf+1,(p+1)*nc) # To store the multiscale basis functions
  K, L, Λ = get_saddle_point_problem(fspace, D, p, nc)
  f1 = zeros(T,size(K,1))
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
    lmat_el = L[freenodes,gn]
    loadvec_el = (f1 - K[:,bnodes]*bvals)
    for _=1:p+1
      fvecs_el = [loadvec_el[freenodes]; Λ[gn, index]]
      lhs = [stima_el lmat_el; (lmat_el)' spzeros(T, length(gn), length(gn))]
      rhs = fvecs_el           
      sol = lhs\rhs                 
      basis_vec_ms[freenodes,index] = sol[1:length(freenodes)]
      index += 1   
    end
  end
  basis_vec_ms
end

#=
For applying non homogeneous boundary condition. Equuivalent to solving the following problem

Find ̃Λᵧ ∈ H¹₀(Nˡ(K)), λᵧ ∈ Vₕᴾ(K) s.t
  a(̃Λᵧ, v) + (λᵧ, v) = -a(gₕ, v)
  (̃Λᵧ, μ) = 0
for all v ∈ H¹₀(Nˡ(K)), μ ∈ Vₕᴾ(K) on the boundary elements only i.e, K ∩ ∂Ω ≠ ∅

The function gₕ is defined as gₕ ∈ P₁(𝒯ₕ) with 
  gₕ(z) = g(z) ∀ z ∈ Γd,
  gₕ(z) = 0 ∀ z ∉ Γd.
=#
"""
Compute the boundary  projection matrix
"""
function compute_boundary_correction_matrix(fspace::FineScaleSpace, D::Function, p::Int64, nc::Int64, l::Int64,
  patch_indices_to_global_indices::Vector{AbstractVector{Int64}}; T=Float64)
  # Compute the projection (solve the saddle point problems) only on the boundary elements
  boundary_elems = [1,nc]
  n_boundary_elems = 1:length(boundary_elems)
  # Begin solving the problem
  nf = fspace.nf
  q = fspace.q
  start, last = max(1,1-l), min(nc,1+l)
  dims = ((last-start+1)*(p+1)+2) # ( ((l+1)*(p+1) = No of patch elements) + (2 = Boundary contribution of stima))
  boundary_correction = spzeros(T, q*nf+1, dims*length(boundary_elems)) # 2 patch elements
  K, L, _ = get_saddle_point_problem(fspace, D, p, nc)
  # Begin solving 
  for (t,i) in zip(boundary_elems,n_boundary_elems)
    tn = patch_indices_to_global_indices[t]
    bn = [tn[1],tn[end]]
    fn = setdiff(tn, bn)    
    start = max(1,t-l)
    last = min(nc,t+l)
    gn = start*(p+1)-p:last*(p+1)    
    lhs = -[K[fn,fn] L[fn,gn]; L[fn,gn]' spzeros(T,(last-start+1)*(p+1),(last-start+1)*(p+1))]
    # Boundary contributions of the LHS
    rhs = collect([K[fn,bn] zero(L[fn,gn]); (zero(L[bn,gn]))' spzeros(T, (last-start+1)*(p+1), (last-start+1)*(p+1))])
    # Invert to compute the projection matrix
    boundary_correction[fn,(dims)*i-(dims-1):dims*i] = (lhs\rhs)[1:length(fn),:] 
  end
  boundary_correction
end
"""
Apply the boundary projection matrix to the Dirichlet boundary condition
"""
function apply_boundary_correction(BC::SparseMatrixCSC{Float64,Int64}, bnodes::Vector{Int64}, bvals::Vector{Float64}, 
  patch_indices_to_global_indices::Vector{AbstractVector{Int64}}, p::Int64, nc::Int64, l::Int64, fspace::FineScaleSpace)
  nf = fspace.nf
  q = fspace.q
  boundary_elems = [1,nc]
  n_boundary_elems = 1:length(boundary_elems)
  boundary_correction = zeros(Float64, q*nf+1) # Zero vector to store the result  
  _bv(bvals, i) = (i==1) ? [bvals[1], 0.0] : [0.0, bvals[2]]
  start, last = max(1,1-l), min(nc,1+l)
  dims = ((last-start+1)*(p+1)+2) # ( ((l+1)*(p+1) = No of patch elements) + (2 = Boundary contribution of stima))
  bvec_el = zeros(Float64, dims)
  # Compute the boundary correction
  for (t,i) in zip(boundary_elems,n_boundary_elems)
    tn = patch_indices_to_global_indices[t]
    bn = [tn[1], tn[end]]
    fn = setdiff(tn,bn)
    bv = _bv(bvals, i)
    bvec_el[n_boundary_elems] = bv
    boundary_correction[fn] += BC[fn,(dims)*i-(dims-1):dims*i]*bvec_el # Compute the projection of the ith DBC value
  end
  boundary_correction[bnodes] = bvals # Fill in the DBC values
  boundary_correction
end