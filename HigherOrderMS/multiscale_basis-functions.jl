#### ##### ##### ##### ##### ##### ##### ##### 
# Module containing the functions to obtain: #           
# 1) the multiscale bases.                   #
# 2) the matrix-vector contributions.        #
#### ##### ##### ##### ##### ##### ##### #####
function compute_ms_basis(domain::Tuple{Float64,Float64}, D::Function, f::Function, fespaces::Tuple{Int64,Int64},
  nels::Tuple{Int64,Int64}, l::Int64, fullnodes::Vector{AbstractVector{Int64}}, qorder::Int64,
  coarse_indices_to_fine_indices::Vector{AbstractVector{Int64}}, ms_elem::Vector{Vector{Int64}})
  q,p = fespaces 
  nf,nc = nels
  basis_vec_ms = spzeros(Float64,q*nf+1,(p+1)*nc) # To store the multiscale basis functions
  KMf, lmat, f2, U = get_saddle_point_problem(domain, D, f, fespaces, nels, qorder)
  stima, massma, loadvec = KMf
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
  basis_elem_ms = lazy_map(Reindex(basis_vec_ms), coarse_indices_to_fine_indices, ms_elem); 
  basis_elem_ms_t = lazy_map(transpose, basis_elem_ms);
  B = cache(basis_elem_ms);
  Bt = cache(basis_elem_ms_t);
  (stima, massma, loadvec), (basis_vec_ms,B,Bt), U
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
