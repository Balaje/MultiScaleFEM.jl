##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #
#  Functions to assemble the stiffness, mass and multiscale matrix vector system   #
##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #

"""
Function to assemble the stiffness matrix
"""
function assemble_stiffness_matrix!(cache, D::Function, u!::Function, v!::Function, J_exp::Int64)
  quad_data, matData,elems, bases, M = cache  
  # Unpack all the data
  xqs, Dxqs, J, quad = quad_data
  iiM, jjM, vvM = matData    
  elem_i, elem_j = elems
  bu, bv = bases
  nel = size(elem_i,1)
  p = size(elem_i,2) - 1
  q = size(elem_j,2) - 1
  qs, ws = quad
  # Fill up the matrices
  qrule = size(xqs, 2)
  map!(D, Dxqs, xqs)
  fill!(M,0.0)
  fill!(vvM,0.0)
  for k=1:qrule
    xq = qs[k]
    wq = ws[k]
    u!(bu, xq)
    v!(bv, xq)      
    index = 0
    for i=1:p+1, j=1:q+1        
      @turbo for l=1:nel
        iiM[index+l] = elem_i[l, i]
        jjM[index+l] = elem_j[l, j]
        vvM[index+l] += Dxqs[l,k]*(J[l])^J_exp*bu[3][i]*bv[3][j]*wq                              
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
function assemble_load_vector!(cache, f::Function, u!::Function, J_exp::Int64)
  quad_data, vecData, h1elem, bu, F = cache
  # Unpack all the data
  xqs, fxqs, J, quad = quad_data
  iiV, vvV = vecData
  elem_i = h1elem
  nel = size(elem_i,1)
  qs, ws = quad
  q = size(elem_i,2) - 1
  # Fill up the matrices
  qrule = size(xqs,2)
  map!(f, fxqs, xqs)
  fill!(F,0.0)
  fill!(vvV,0.0)
  for k=1:qrule
    xq = qs[k]
    wq = ws[k]
    u!(bu, xq)
    index = 0
    for i=1:q+1
      @turbo for l=1:nel
        iiV[index+l] = elem_i[l,i]
        vvV[index+l] += fxqs[l,k]*(J[l])^J_exp*bu[3][i]*wq
      end
      index+=nel
    end
  end
  @turbo for k=1:lastindex(iiV)
    F[iiV[k]] += vvV[k]
  end
  F
end
function get_quad_data(nds::AbstractVector{Float64}, elem::Matrix{Int64}, 
  quad::Tuple{Vector{Float64}, Vector{Float64}})
  nds_elem = nds[elem]
  qs = quad[1]
  nel = size(elem,1)
  xqs = Matrix{Float64}(undef, nel, length(qs))
  J = (nds_elem[:,2]-nds_elem[:,1])*0.5
  for i=1:lastindex(qs)
    xqs[:,i] = (nds_elem[:,2]+nds_elem[:,1])*0.5 + (nds_elem[:,2]-nds_elem[:,1])*0.5*qs[i]
  end
  Dxqs = zero(xqs)
  xqs, Dxqs, J, quad
end
function stiffness_matrix_cache(nds::AbstractVector{Float64}, elem::Matrix{Int64}, 
  quad::Tuple{Vector{Float64}, Vector{Float64}}, q::Int64)
  nel = size(elem,1)
  iiM = Vector{Int64}(undef, (q+1)^2*nel)
  jjM = Vector{Int64}(undef, (q+1)^2*nel)
  vvM = Vector{Float64}(undef, (q+1)^2*nel)
  fill!(iiM,0); fill!(jjM,0); fill!(vvM,0.0) 
  quad_data = get_quad_data(nds, elem, quad)
  bc = lagrange_basis_cache(q)
  h1elem =  [q*i+j-(q-1) for i = 1:nel, j=0:q]
  assem_mat = spzeros(Float64, q*nel+1, q*nel+1)
  quad_data, (iiM, jjM, vvM), (h1elem, h1elem), (bc, bc), assem_mat
end
function load_vector_cache(nds::AbstractVector{Float64}, elem::Matrix{Int64}, 
  quad::Tuple{Vector{Float64}, Vector{Float64}}, q::Int64)
  nel = size(elem,1)
  iiV = Vector{Int64}(undef,(q+1)*nel)
  vvV = Vector{Float64}(undef,(q+1)*nel)
  fill!(iiV,0); fill!(vvV,0.0)
  quad_data = get_quad_data(nds, elem, quad)
  bc = lagrange_basis_cache(q)
  h1elem =  [q*i+j-(q-1) for i = 1:nel, j=0:q]
  assem_vec = zeros(Float64, q*nel+1)
  quad_data, (iiV, vvV), h1elem, bc, assem_vec
end

"""
Function to assemble the rectangular matrix arising due to the Lagrange multiplier.  
"""
function assemble_lm_matrix!(cache, Λ!::Function, u!::Function, J_exp::Int64)
  assem_cache, l_mat_cache = cache
  nds_elem, L, lb_cache = l_mat_cache
  nc = size(nds_elem,1)    
  p = length(lb_cache)-1
  index = 1
  for t=1:nc
    nds = (nds_elem[t,1], nds_elem[t,2])
    for tt=1:p+1    
      l(y) = Λ!(lb_cache, y, nds)[tt]
      assemble_load_vector!(assem_cache, l, u!, J_exp)
      F = assem_cache[5]
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
  quad::Tuple{Vector{Float64}, Vector{Float64}}, fespaces::Tuple{Int64,Int64})
  elem_coarse, elem_fine = elem
  nds_coarse, nds_fine = nds
  nf = size(elem_fine,1)
  nc = size(elem_coarse,1)
  q,p = fespaces
  assem_cache = load_vector_cache(nds_fine, elem_fine, quad, q)
  assem_lm_mat = spzeros(Float64, q*nf+1, (p+1)*nc)
  assem_cache, (nds_coarse[elem_coarse], assem_lm_mat, legendre_basis_cache(p))
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

"""
Function to assemble the multiscale stiffness matrix
"""
function fillsKms!(cache, local_basis_vecs::Matrix{Float64}, coarse_elem_indices_to_fine_elem_indices::AbstractVector{Int64}, Kₛ::AbstractMatrix{Float64})  
  C, Ct, Lt, _ = cache 
  L = @views local_basis_vecs[coarse_elem_indices_to_fine_elem_indices, :]
  transpose!(Lt, L)
  mul!(Ct, Kₛ, L)
  mul!(C, Lt, Ct)  
  C
end

function fillsFms!(cache, local_basis_vecs::Matrix{Float64}, coarse_elem_indices_to_fine_elem_indices::AbstractVector{Int64}, F::AbstractVector{Float64})
  _,_, Lt, res = cache
  L = @views local_basis_vecs[coarse_elem_indices_to_fine_elem_indices, :]
  transpose!(Lt, L)
  mul!(res, Lt, F)
  res
end

function assemble_ms_matrix!(cache, sKms, ms_elem::Vector{Vector{Int64}})
  nc = size(ms_elem,1)
  K = cache
  fill!(K,0.0)
  for t=1:nc
    local_dof = size(ms_elem[t],1)
    elem = ms_elem[t]
    local_mat = sKms[t]
    @turbo for ti=1:local_dof, tj=1:local_dof
      K[elem[ti], elem[tj]] += local_mat[ti,tj]
    end
  end
end

function assemble_ms_vector!(cache, sFms, ms_elem::Vector{Vector{Int64}})
  nc = size(ms_elem,1)
  F = cache
  fill!(F,0.0)
  for t=1:nc
    local_dof = size(ms_elem[t],1)
    elem = ms_elem[t]
    local_vec = sFms[t]
    @turbo for ti=1:local_dof
      F[elem[ti]] += local_vec[ti]  
    end
  end
end

function build_solution!(cache, sol::Vector{Float64}, local_basis_vecs::Vector{Matrix{Float64}})  
  nc = size(local_basis_vecs, 1)
  p = size(local_basis_vecs[1], 2)-1
  res, sol_cache = cache
  fill!(sol_cache,0.0)
  fill!(res, 0.0)
  for j=1:nc, i=0:p
    get_local_basis!(sol_cache, local_basis_vecs, j, 1:length(sol_cache), i+1)
    @turbo for tt=1:lastindex(res)
      res[tt] += sol[(p+1)*j+i-p]*sol_cache[tt]
    end
  end
end