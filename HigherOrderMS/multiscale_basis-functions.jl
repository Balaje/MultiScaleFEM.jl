#### ##### ##### ##### ##### ##### ##### ##### ##### ###
# Module containing the MultiScaleBases implementation #
#### ##### ##### ##### ##### ##### ##### ##### ##### ###
struct ms_basis_cache{A,B}
  stima::SparseMatrixCSC{A,B}
  lmat::SparseMatrixCSC{A,B}
  zvals::Vector{A}
  fvecs::Diagonal{A, Vector{A}}
end
function ms_basis_cache(
  matcache::Tuple{stiffness_matrix_cache{Int64,Float64}, 
  Tuple{Matrix{load_vector_cache{Int64, Float64}}, Tuple{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}}},
  Diagonal{Float64,Vector{Float64}}}, 
  nf::Int64, 
  fespaces::Tuple{Int64,Int64})
  q,p = fespaces
  stima_cache, lmat_cache, fvecs_cache = matcache
  stima = assemble_stiffness_matrix!(stima_cache, -1)
  lmat = assemble_lm_matrix!(lmat_cache, 1)
  fvecs = assemble_lm_l2_matrix!(fvecs_cache, nds_coarse, elem_coarse, p)
  ms_basis_cache{Float64,Int64}(stima, lmat, zeros(Float64,q*nf+1), fvecs)
end

function compute_ms_basis(cache::ms_basis_cache{Float64,Int64}, fespaces::Tuple{Int64,Int64}, l::Int64, 
  nels::Tuple{Int64,Int64}, fullnodes::Vector{AbstractVector{Int64}})
  q,p = fespaces 
  nf,nc = nels
  basis_vec_ms = spzeros(Float64,q*nf+1,(p+1)*nc) # To store the multiscale basis functions
  stima = cache.stima 
  lmat = cache.lmat
  f1 = cache.zvals
  f2 = cache.fvecs
  index = 1;
  for t=1:nc
    fn = fullnodes[t][2:end-1]
    start = max(1,t-l)
    last = min(nc,t+l)
    gn = start*(p+1)-p:last*(p+1)    
    stima_el = stima[fn,fn]
    lmat_el = lmat[fn,gn]
    for _=1:p+1
      fvecs_el = @views [f1[fn]; f2[gn, index]]
      lhs = [stima_el lmat_el; (lmat_el)' spzeros(Float64, length(gn), length(gn))]
      rhs = fvecs_el           
      sol = lhs\rhs                 
      basis_vec_ms[fn,index] = sol[1:length(fn)]
      index += 1   
    end
  end
  basis_vec_ms
end

function contrib_cache(nds::AbstractVector{Float64}, coarse_elem_indices_to_fine_elem_indices::Vector{AbstractVector{Int64}}, 
  quad::Tuple{Vector{Float64}, Vector{Float64}}, D::Function, f::Function, q::Int64)
  nc = size(coarse_elem_indices_to_fine_elem_indices,1)
  mat_contrib_cache = Vector{stiffness_matrix_cache{Int64,Float64}}(undef,nc)  
  vec_contrib_cache = Vector{load_vector_cache{Int64,Float64}}(undef,nc)  
  for t=1:nc
    el_conn_el = [i+j for i=1:length(coarse_elem_indices_to_fine_elem_indices[t])-1, j=0:1]
    nds_el = nds[coarse_elem_indices_to_fine_elem_indices[t]]
    mat_contrib_cache[t] = stiffness_matrix_cache(nds_el, el_conn_el, quad, D, ∇φᵢ!, q)
    vec_contrib_cache[t] = load_vector_cache(nds_el, el_conn_el, quad, f, φᵢ!, q)
  end
  mat_contrib_cache, vec_contrib_cache
end