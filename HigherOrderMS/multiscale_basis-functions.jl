#### ##### ##### ##### ##### ##### ##### ##### ##### ###
# Module containing the MultiScaleBases implementation #
#### ##### ##### ##### ##### ##### ##### ##### ##### ###

function compute_ms_bases!(cache::Tuple{Tuple{SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{Float64, Int64}}, Tuple{Vector{Float64}, Diagonal{Float64, Vector{Float64}}}, Tuple{Vector{SparseMatrixCSC{Float64, Int64}}, Vector{UnitRange{Int64}}}}, p::Int64, l::Int64)
  mats, fvecs, bases_data = cache
  f1, f2 = fvecs
  stima, lmat = mats
  basis_vec_ms, fns = bases_data
  nc = size(basis_vec_ms,1)
  index = 1
  for t=1:nc
    start = max(1,t-l)
    last = min(nc,t+l)
    fn = fns[t]
    gn = start*(p+1)-p:last*(p+1)    
    stima_el = stima[fn,fn]
    lmat_el = lmat[fn,gn]
    for tt=1:p+1
      fvecs_el = @views [f1[fn]; f2[gn, index]]
      lhs = [stima_el lmat_el; (lmat_el)' spzeros(Float64, length(gn), length(gn))]
      rhs = fvecs_el           
      sol = lhs\rhs                 
      basis_vec_ms[t][fn,tt] = sol[1:length(fn)]
      index += 1   
    end
  end
  basis_vec_ms
end

function ms_basis_cache!(matcache::Tuple{Tuple{Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Tuple{Vector{Float64}, Vector{Float64}}},
   Tuple{Vector{Int64}, Vector{Int64}, Vector{Float64}}, Tuple{Matrix{Int64}, Matrix{Int64}}, 
   Tuple{Tuple{Adjoint{Float64, Matrix{Float64}}, Vector{Float64}, Vector{Float64}}, Tuple{Adjoint{Float64, Matrix{Float64}}, Vector{Float64}, Vector{Float64}}}, 
   SparseMatrixCSC{Float64, Int64}}, Tuple{Tuple{Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Tuple{Vector{Float64}, Vector{Float64}}}, Tuple{Vector{Int64}, Vector{Float64}}, Matrix{Int64}, Tuple{Adjoint{Float64, Matrix{Float64}}, Vector{Float64}, Vector{Float64}}, Vector{Float64}}, Tuple{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Vector{Float64}}}, 
   Diagonal{Float64, Vector{Float64}}}, D::Function, nf::Int64, fespaces::Tuple{Int64,Int64}, 
  basis_vec_ms::Vector{SparseMatrixCSC{Float64,Int64}}, patch_to_fine_scale::Vector{AbstractVector{Int64}})
  q,p = fespaces
  stima_cache, lmat_cache, fvecs_cache = matcache
  stima = assemble_stiffness_matrix!(stima_cache, D, ∇φᵢ!, ∇φᵢ!, -1)
  lmat = assemble_lm_matrix!(lmat_cache, Λₖ!, φᵢ!, 1)
  fvecs = assemble_lm_l2_matrix!(fvecs_cache, nds_coarse, elem_coarse, p)
  fns = [gn[2:length(gn)-1] for gn in patch_to_fine_scale]
  (stima, lmat), (zeros(Float64, q*nf+1), fvecs), (basis_vec_ms, fns)
end

function sort_basis_vectors!(sorted_basis::Vector{SparseMatrixCSC{Float64,Int64}}, basis_vec_ms::Vector{SparseMatrixCSC{Float64,Int64}}, ms_elem::Vector{Vector{Int64}}, p::Int64, l::Int64)
  nc = size(ms_elem,1)
  for t=1:nc
    start = max(1, t-l)
    last = min(nc, t+l)
    binds = start:last
    nd = (last-start+1)*(p+1)
    for j=1:nd
      ii1 = ceil(Int,j/(p+1))
      ll1 = ceil(Int,(j-1)%(p+1)) + 1
      sorted_basis[t][:,j] = basis_vec_ms[binds[ii1]][:,ll1]
    end
  end
  sorted_basis
end

function get_local_basis!(cache, fullvec::Vector{SparseMatrixCSC{Float64,Int64}}, el::Int64, fn::AbstractVector{Int64}, ind::Int64)
  @assert length(cache) == length(fn)
  lbv = fullvec[el]
  copyto!(cache, view(lbv, fn, ind))
end

"""
Function to extract the stiffness matrices element wise
"""
function mat_contribs!(assem_cache, D::Function, u!::Function, v!::Function, J_exp::Int64)
  assemble_stiffness_matrix!(assem_cache, D, u!, v!, J_exp)
  get_stiffness_matrix_from_cache(assem_cache)
end
"""
Function to extract the load vector element wise
"""
function vec_contribs!(assem_cache, f::Function, u!::Function, J_exp::Int64)
  assemble_load_vector!(assem_cache, f, u!, J_exp)
  get_load_vector_from_cache(assem_cache)
end
function contrib_cache(nds::AbstractVector{Float64}, coarse_elem_indices_to_fine_elem_indices::Vector{AbstractVector{Int64}}, 
  quad::Tuple{Vector{Float64}, Vector{Float64}}, q::Int64)
  nc = size(coarse_elem_indices_to_fine_elem_indices,1)
  mat_contrib_cache = Vector{Any}(undef,nc)  
  vec_contrib_cache = Vector{Tuple{Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Tuple{Vector{Float64}, Vector{Float64}}}, Tuple{Vector{Int64}, Vector{Float64}}, 
  Matrix{Int64}, Tuple{Adjoint{Float64, Matrix{Float64}}, Vector{Float64}, Vector{Float64}}, Vector{Float64}}}(undef,nc)  
  for t=1:nc
    el_conn_el = [i+j for i=1:length(coarse_elem_indices_to_fine_elem_indices[t])-1, j=0:1]
    nds_el = nds[coarse_elem_indices_to_fine_elem_indices[t]]
    mat_contrib_cache[t] = stiffness_matrix_cache(nds_el, el_conn_el, quad, q)
    vec_contrib_cache[t] = load_vector_cache(nds_el, el_conn_el, quad, q)
  end
  mat_contrib_cache, vec_contrib_cache
end
get_stiffness_matrix_from_cache(cache) = cache[5]
get_load_vector_from_cache(cache) = cache[5]