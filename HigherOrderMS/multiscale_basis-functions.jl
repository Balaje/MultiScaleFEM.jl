#### ##### ##### ##### ##### ##### ##### ##### 
# Module containing the functions to obtain: #           
# 1) the multiscale bases.                   #
# 2) the matrix-vector contributions.        #
#### ##### ##### ##### ##### ##### ##### #####
function compute_ms_basis(domain::Tuple{Float64,Float64}, D::Function, f::Function, fespaces::Tuple{Int64,Int64},
  nels::Tuple{Int64,Int64}, l::Int64, patch_indices_to_global_indices::Vector{AbstractVector{Int64}}, qorder::Int64,
  bnodes::Vector{Int64}, bvals::Vector{Float64})
  q,p = fespaces 
  nf,nc = nels
  basis_vec_ms = spzeros(Float64,q*nf+1,(p+1)*nc) # To store the multiscale basis functions
  K, L, Λ = get_saddle_point_problem(domain, D, f, fespaces, nels, qorder)
  f1 = zeros(Float64,size(K,1))
  index = 1
  for t=1:nc
    fullnodes = patch_indices_to_global_indices[t]
    freenodes = setdiff(fullnodes, bnodes)
    start = max(1,t-l)
    last = min(nc,t+l)
    gn = start*(p+1)-p:last*(p+1)    
    stima_el = K[freenodes,freenodes]
    lmat_el = L[freenodes,gn]
    loadvec_el = (f1 - K[:,bnodes]*bvals)
    for _=1:p+1
      fvecs_el = [loadvec_el[freenodes]; Λ[gn, index]]
      lhs = [stima_el lmat_el; (lmat_el)' spzeros(Float64, length(gn), length(gn))]
      rhs = fvecs_el           
      sol = lhs\rhs                 
      basis_vec_ms[freenodes,index] = sol[1:length(freenodes)]
      basis_vec_ms[bnodes,index] = bvals
      index += 1   
    end
  end
  basis_vec_ms
end

function mat_contribs(stima::SparseMatrixCSC{Float64,Int64}, coarse_indices_to_fine_indices::AbstractVector{Int64}, t::Int64, nc::Int64)::SparseMatrixCSC{Float64,Int64}
  stima_el = getindex(stima, coarse_indices_to_fine_indices, coarse_indices_to_fine_indices)
  (t!=1) && (stima_el[1,1]/=2.0)
  (t!=nc) && (stima_el[end,end]/=2.0)
  stima_el
end
function vec_contribs(loadvec::Vector{Float64}, coarse_indices_to_fine_indices::AbstractVector{Int64}, t::Int64, nc::Int64)::Vector{Float64}
  loadvec_el = getindex(loadvec, coarse_indices_to_fine_indices)
  (t!=1) && (loadvec_el[1]/=2.0)
  (t!=nc) && (loadvec_el[end]/=2.0)
  loadvec_el
end
