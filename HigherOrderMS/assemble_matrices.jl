##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #
#  Functions to assemble the stiffness, mass and multiscale matrix vector system   #
##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #

function get_quad_data(nds::AbstractVector{Float64}, elem::Matrix{Int64}, quad::Tuple{Vector{Float64}, Vector{Float64}}, 
  f::Function, u!::Function, bc::Tuple{Adjoint{Float64, Matrix{Float64}}, Vector{Float64}, Vector{Float64}})
  nds_elem = nds[elem]
  qs = quad[1]
  nel = size(elem,1)
  xqs = Matrix{Float64}(undef, nel, length(qs))
  basis_quad = Vector{Vector{Float64}}(undef, length(qs))
  J = (nds_elem[:,2]-nds_elem[:,1])*0.5
  for i=1:lastindex(qs)
    xqs[:,i] = (nds_elem[:,2]+nds_elem[:,1])*0.5 + (nds_elem[:,2]-nds_elem[:,1])*0.5*qs[i]
    basis_quad[i] = u!(bc, qs[i])
  end
  fxqs = map(f, xqs)
  xqs, fxqs, J, quad, basis_quad
end
"""
Function to assemble the stiffness matrix
"""
struct stiffness_matrix_cache{A,B}
  quad_data::Tuple{Matrix{B}, Matrix{B}, Vector{B}, Tuple{Vector{B}, Vector{B}}, Vector{Vector{B}}}
  mat_data::Tuple{Vector{A}, Vector{A}, Vector{B}}
  elem::Tuple{Matrix{A}, Matrix{A}}
  assem_mat::SparseMatrixCSC{B,A}
end
function stiffness_matrix_cache(nds::AbstractVector{Float64}, elem::Matrix{Int64}, 
  quad::Tuple{Vector{Float64}, Vector{Float64}}, D::Function, u!::Function, q::Int64)
  nel = size(elem,1)
  iiM = Vector{Int64}(undef, (q+1)^2*nel)
  jjM = Vector{Int64}(undef, (q+1)^2*nel)
  vvM = Vector{Float64}(undef, (q+1)^2*nel)
  fill!(iiM,0); fill!(jjM,0); fill!(vvM,0.0) 
  bc = lagrange_basis_cache(q)
  quad_data = get_quad_data(nds, elem, quad, D, u!, bc)
  h1elem =  [q*i+j-(q-1) for i = 1:nel, j=0:q]
  assem_mat = spzeros(Float64, q*nel+1, q*nel+1)
  stiffness_matrix_cache{Int64,Float64}(quad_data, (iiM, jjM, vvM), (h1elem, h1elem), assem_mat)
end
function assemble_stiffness_matrix!(cache::stiffness_matrix_cache{Int64,Float64}, J_exp::Int64)
  #quad_data, matData,elems, bases, M = cache  
  quad_data = cache.quad_data
  matData = cache.mat_data
  elems = cache.elem
  M = cache.assem_mat
  # Unpack all the data
  xqs, Dxqs, J, quad, basis_quad = quad_data
  iiM, jjM, vvM = matData    
  elem_i, elem_j = elems
  nel = size(elem_i,1)
  p = size(elem_i,2) - 1
  q = size(elem_j,2) - 1
  ws = quad[2]
  # Fill up the matrices
  qrule = size(xqs, 2)
  fill!(M,0.0)
  fill!(vvM,0.0)
  for k = 1:qrule
    wq = ws[k]    
    index = 0
    for i=1:p+1, j=1:q+1        
      @turbo for l=1:nel
        iiM[index+l] = elem_i[l, i]
        jjM[index+l] = elem_j[l, j]
        vvM[index+l] += Dxqs[l,k]*(J[l])^J_exp*basis_quad[k][i]*basis_quad[k][j]*wq                              
      end
      index+=nel
    end
  end    
  @simd for k=1:lastindex(iiM)
    @inbounds M[iiM[k], jjM[k]] += vvM[k]
  end
  M
end
"""
Function to assemble the load vector
"""
struct load_vector_cache{A,B} 
  quad_data::Tuple{Matrix{B}, Matrix{B}, Vector{B}, Tuple{Vector{B}, Vector{B}}, Vector{Vector{B}}}
  mat_data::Tuple{Vector{A}, Vector{B}}
  elem::Matrix{A}
  assem_vec::Vector{B}
end
function load_vector_cache(nds::AbstractVector{Float64}, elem::Matrix{Int64}, 
  quad::Tuple{Vector{Float64}, Vector{Float64}}, f::Function, u!::Function, q::Int64)
  nel = size(elem,1)
  iiV = Vector{Int64}(undef,(q+1)*nel)
  vvV = Vector{Float64}(undef,(q+1)*nel)
  fill!(iiV,0); fill!(vvV,0.0)
  bc = lagrange_basis_cache(q)
  quad_data = get_quad_data(nds, elem, quad, f, u!, bc)
  h1elem =  [q*i+j-(q-1) for i = 1:nel, j=0:q]
  assem_vec = zeros(Float64, q*nel+1)
  load_vector_cache{Int64, Float64}(quad_data, (iiV, vvV), h1elem, assem_vec)
end
function assemble_load_vector!(cache::load_vector_cache{Int64,Float64}, J_exp::Int64)
  quad_data = cache.quad_data
  vecData = cache.mat_data
  h1elem = cache.elem
  F = cache.assem_vec
  # Unpack all the data
  xqs, fxqs, J, quad, basis_quad = quad_data
  iiV, vvV = vecData
  elem_i = h1elem
  nel = size(elem_i,1)
  qs, ws = quad
  q = size(elem_i,2) - 1
  # Fill up the matrices
  qrule = size(xqs,2)
  fill!(F,0.0)
  fill!(vvV,0.0)
  for k=1:qrule
    wq = ws[k]
    index = 0
    for i=1:q+1
      @turbo for l=1:nel
        iiV[index+l] = elem_i[l,i]
        vvV[index+l] += fxqs[l,k]*(J[l])^J_exp*basis_quad[k][i]*wq
      end
      index+=nel
    end
  end
  @turbo for k=1:lastindex(iiV)
    F[iiV[k]] += vvV[k]
  end
  F
end

"""
Function to assemble the rectangular matrix arising due to the Lagrange multiplier.  
"""
function assemble_lm_matrix!(cache, J_exp::Int64)
  assem_cache, l_mat_cache = cache
  nds_elem, L = l_mat_cache
  nc = size(nds_elem,1)    
  p = Int(size(L,2)/nc)-1
  index = 1
  for t=1:nc   
    for tt=1:p+1    
      F = assemble_load_vector!(assem_cache[t,tt], J_exp)
      @inbounds @simd for k=1:lastindex(F)
        L[k,index] = F[k]
      end
      index = index+1
    end
  end
  L
end
function lm_matrix_cache(nds::Tuple{AbstractVector{Float64}, AbstractVector{Float64}}, 
  elem::Tuple{Matrix{Int64}, Matrix{Int64}},
  quad::Tuple{Vector{Float64}, Vector{Float64}}, fespaces::Tuple{Int64,Int64}, Λ!::Function, u!::Function)
  elem_coarse, elem_fine = elem
  nds_coarse, nds_fine = nds
  nf = size(elem_fine,1)
  nc = size(elem_coarse,1)
  q,p = fespaces
  assem_cache = Matrix{load_vector_cache{Int64,Float64}}(undef, nc, (p+1))
  lb_cache = legendre_basis_cache(p)
  for t=1:nc
    nds = (nds_coarse[elem_coarse[t,1]], nds_coarse[elem_coarse[t,2]])
    for tt=1:p+1
      l(y) = Λ!(lb_cache, y, nds)[tt]
      assem_cache[t,tt] = load_vector_cache(nds_fine, elem_fine, quad, l, u!, q)
    end
  end
  assem_lm_mat = spzeros(Float64, q*nf+1, (p+1)*nc)
  assem_cache, (nds_coarse[elem_coarse], assem_lm_mat)
end

"""
Function to assemble the matrix associated with the RHS of the saddle point problem
"""
function assemble_lm_l2_matrix!(l2mat, nds::AbstractVector{Float64}, elem::Matrix{Int64}, p::Int64)   
  nel = size(elem,1)
  index = 1
  for t=1:nel
    h = nds[elem[t,2]] - nds[elem[t,1]]
    @simd for i=1:p+1
      @inbounds l2mat[(index-1)+i, (index-1)+i] = (h/(2*(i-1)+1))
    end
    index = index + (p+1)
  end
  l2mat
end
function lm_l2_matrix_cache(nc::Int64, p::Int64)
  Diagonal(ones(Float64,nc*(p+1)))
end

function assemble_ms_matrix(ms_elem_mats, ms_elem::Vector{Vector{Int64}}, nc::Int64, p::Int64)
  IJV = BroadcastVector(findnz, ms_elem_mats);
  IJV1 = BroadcastVector(repeat, ms_elem, BroadcastVector(length, ms_elem))
  IJV2 = BroadcastVector(vec, BroadcastVector(repeat, BroadcastVector(transpose,ms_elem), BroadcastVector(length, ms_elem)));
  IJV3 = BroadcastVector(getindex, IJV, 3);
  M = BroadcastVector(sparse, IJV1, IJV2, IJV3, nc*(p+1), nc*(p+1));
  stima_ms = applied(sum, applied(Tuple, M))
  materialize(stima_ms)
end
function assemble_ms_vector(ms_elem_vecs, ms_elem::Vector{Vector{Int64}}, nc::Int64, p::Int64)
  V = BroadcastVector(sparsevec, ms_elem, ms_elem_vecs, nc*(p+1))
  loadvec_ms = applied(sum, applied(Tuple, V))
  collect(materialize(loadvec_ms))
end

function build_solution!(cache, sol::Vector{Float64}, basis_vecs::SparseMatrixCSC{Float64,Int64})  
  nc = size(sol, 1)
  p = Int(size(basis_vecs, 2)/nc)-1
  res, sol_cache = cache
  fill!(sol_cache,0.0)
  fill!(res, 0.0)
  index = 1
  for j=1:nc, i=0:p
    @simd for tt=1:lastindex(res)
      @inbounds res[tt] += sol[(p+1)*j+i-p]*basis_vecs[tt,index]      
    end
    index+=1
  end
end