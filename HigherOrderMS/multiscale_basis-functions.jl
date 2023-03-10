#### ##### ##### ##### ##### ##### ##### ##### ##### ###
# Module containing the MultiScaleBases implementation #
#### ##### ##### ##### ##### ##### ##### ##### ##### ###
function compute_ms_basis(domain::Tuple{Float64,Float64}, D::Function, f::Function, fespaces::Tuple{Int64,Int64},
  nels::Tuple{Int64,Int64}, l::Int64, fullnodes::Vector{AbstractVector{Int64}}, qorder::Int64)
  q,p = fespaces 
  nf,nc = nels
  basis_vec_ms = spzeros(Float64,q*nf+1,(p+1)*nc) # To store the multiscale basis functions
  Kf, lmat, f2 = get_saddle_point_problem(domain, D, f, fespaces, nels, qorder)
  stima, loadvec = Kf
  f1 = zero(loadvec)
  index = 1
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
  (stima, loadvec), basis_vec_ms
end

# function contrib_cache(nds::AbstractVector{Float64}, coarse_elem_indices_to_fine_elem_indices::Vector{AbstractVector{Int64}}, 
#   quad::Tuple{Vector{Float64}, Vector{Float64}}, D::Function, f::Function, q::Int64)
#   nc = size(coarse_elem_indices_to_fine_elem_indices,1)
#   mat_contrib_cache = Vector{stiffness_matrix_cache{Int64,Float64}}(undef,nc)  
#   vec_contrib_cache = Vector{load_vector_cache{Int64,Float64}}(undef,nc)  
#   for t=1:nc
#     el_conn_el = [i+j for i=1:length(coarse_elem_indices_to_fine_elem_indices[t])-1, j=0:1]
#     nds_el = nds[coarse_elem_indices_to_fine_elem_indices[t]]
#     mat_contrib_cache[t] = stiffness_matrix_cache(nds_el, el_conn_el, quad, D, ∇φᵢ!, q)
#     vec_contrib_cache[t] = load_vector_cache(nds_el, el_conn_el, quad, f, φᵢ!, q)
#   end
#   mat_contrib_cache, vec_contrib_cache
# end

# function contrib_cache(domain::Tuple{Float64, Float64}, D::Function, f::Function, q::Int64, 
#   coarse_elem_indices_to_fine_elem_indices::Vector{AbstractVector{Int64}}, qorder::Int64)

# end

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
