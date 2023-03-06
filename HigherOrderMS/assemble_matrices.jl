##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #
#  Functions to assemble the stiffness, mass and multiscale matrix vector system   #
##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #

"""
Function to assemble the stiffness matrix
"""
function assemble_matrix!(cache, D::Function, u!::Function, v!::Function, J_exp::Int64)
  quadData = cache[1]
  matData = cache[2]
  elems = cache[4]
  bu, bv = cache[5] 
  quad = cache[6]
  M = cache[7]
  # Unpack all the data
  qs, ws = quad
  xqs, Dxqs, J = quadData
  iiM, jjM, vvM = matData    
  elem_i, elem_j = elems
  nel = size(elem_i,1)
  p = size(elem_i,2) - 1
  q = size(elem_j,2) - 1
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
function assemble_vector!(cache, f::Function, u!::Function, J_exp::Int64)
  quadData = cache[1]
  vecData = cache[3]
  elems = cache[4]
  bu = cache[5][1]
  quad = cache[6]
  F = cache[8]
  # Unpack all the data
  qs, ws = quad
  xqs, fxqs, J = quadData
  iiV, vvV = vecData
  elem_i = elems[1]
  nel = size(elem_i,1)
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
function assembler_cache(nds::AbstractVector{Float64}, elem::Matrix{Int64}, 
  quad::Tuple{Vector{Float64}, Vector{Float64}}, q::Int64)
  nds_elem = nds[elem]
  nel = size(elem,1)
  iiM = Vector{Int64}(undef, (q+1)^2*nel)
  jjM = Vector{Int64}(undef, (q+1)^2*nel)
  vvM = Vector{Float64}(undef, (q+1)^2*nel)
  fill!(iiM,0)
  fill!(jjM,0)
  fill!(vvM,0.0)
  iiV = Vector{Int64}(undef,(q+1)*nel)
  vvV = Vector{Float64}(undef,(q+1)*nel)
  fill!(iiV,0)
  fill!(vvV,0.0)
  qs = quad[1]
  xqs = Matrix{Float64}(undef, nel, length(qs))
  J = (nds_elem[:,2]-nds_elem[:,1])*0.5
  for i=1:lastindex(qs)
    xqs[:,i] = (nds_elem[:,2]+nds_elem[:,1])*0.5 + (nds_elem[:,2]-nds_elem[:,1])*0.5*qs[i]
  end
  Dxqs = similar(xqs)
  fill!(Dxqs,0.0)
  bc = lagrange_basis_cache(q)
  h1elem =  [q*i+j-(q-1) for i = 1:nel, j=0:q]
  assem_mat = spzeros(Float64, q*nel+1, q*nel+1)
  assem_vec = zeros(Float64, q*nel+1)
  (xqs, Dxqs, J), (iiM, jjM, vvM), (iiV, vvV), (h1elem, h1elem), (bc,bc), quad, assem_mat, assem_vec
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
      assemble_vector!(assem_cache, l, u!, J_exp)
      F = assem_cache[8]
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
  assem_cache = assembler_cache(nds_fine, elem_fine, quad, q)
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
    @inbounds @fastmath for i=1:p+1, j=1:p+1
      l2mat[(index-1)+i, (index-1)+j] = (i==j)*(h/(2*(j-1)+1))
    end
    index = index + (p+1)
  end
  l2mat
end
function lm_l2_matrix_cache(nc::Int64, p::Int64)
  zeros(Float64, nc*(p+1), nc*(p+1))
end

"""
Function to assemble the multiscale stiffness matrix
"""
function fillsKms!(cache, nc::Int64, p::Int64, l::Int64)
  sKms, local_basis_vecs, coarse_elem_indices_to_fine_elem_indices, ccs, ip = cache
  L, Lt, ipcache = ip
  for t=1:nc
    start = max(1, t-l)
    last = min(nc, t+l)
    binds = start:last
    nd = (last-start+1)*(p+1)
    gtpi = coarse_elem_indices_to_fine_elem_indices[t]
    for ii=1:nd, jj=1:nd
      fill!(ipcache, 0.0)
      ii1 = ceil(Int,ii/(p+1)); ii2 = ceil(Int,jj/(p+1))
      ll1 = ceil(Int,(ii-1)%(p+1)) + 1; ll2 = ceil(Int,(jj-1)%(p+1)) + 1 
      get_local_basis!(L, local_basis_vecs, binds[ii2], gtpi, ll2)     
      get_local_basis!(Lt, local_basis_vecs, binds[ii1], gtpi, ll1) 
      Kₛ = get_stiffness_matrix_from_cache(ccs[t])
      mul!(ipcache, Kₛ, Lt)
      stima_el = sKms[t]
      @simd for tt=1:lastindex(ipcache)
        @inbounds stima_el[ii,jj] += L[tt]*ipcache[tt]
      end   
    end
  end
  sKms
end

function fillsFms!(cache, nc::Int64, p::Int64, l::Int64)
  sFms, local_basis_vecs, coarse_elem_indices_to_fine_elem_indices, ccs, ip = cache  
  L, Lt, ipcache = ip
  for t=1:nc
    start = max(1,t-l)
    last = min(nc,t+l)
    binds = start:last     
    nd = (last-start+1)*(p+1)    
    gtpi = coarse_elem_indices_to_fine_elem_indices[t] 
    for ii=1:nd
      ii1 = ceil(Int,ii/(p+1));
      ll1 = ceil(Int,(ii-1)%(p+1)) + 1;      
      get_local_basis!(Lt, local_basis_vecs, binds[ii1], gtpi, ll1)
      F = get_load_vector_from_cache(ccs[t])
      load_el = sFms[t]
      @simd for tt=1:lastindex(Lt)
        @inbounds load_el[ii] += F[tt]*Lt[tt]
      end
    end
  end
  sFms
end