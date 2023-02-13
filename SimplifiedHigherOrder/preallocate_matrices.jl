######### ######### ######### ######### ######### ######### ######### ########## 
# New file to preallocate the necessary matrices for the new basis computation #
######### ######### ######### ######### ######### ######### ######### ########## 

H¹Conn(q,i,j) = q*i + j + (q-1)
L²Conn(p,i,j) = (p+1)*i + j - p
elem_conn(i,j) = i+j

function preallocate_matrices(domain::Tuple{Float64,Float64}, 
  nc::Int64, nf::Int64, l::Int64, fespaces::Tuple{Int64,Int64})

  q,p = fespaces
  nds_coarse = LinRange(domain[1], domain[2], nc+1)
  nds_fine = LinRange(domain[1], domain[2], nf+1)
  elem_coarse = [elem_conn(i,j) for i=1:nc, j=0:1]
  elem_fine = [elem_conn(i,j) for i=1:nf, j=0:1]
  assem_H¹H¹ = ([H¹Conn(q,i,j) for _=0:q, j=0:q, i=1:nf],
  [H¹Conn(q,i,j) for j=0:q, _=0:q, i=1:nf], 
  [H¹Conn(q,i,j) for j=0:q, i=1:nf])
  
  aspect_ratio = Int(nf/nc)  
  @assert abs(Int(nf/nc) - (nf/nc)) ≈ 0.0

  # Preallocate some global structures
  basis_vec_patch = Vector{Matrix{Float64}}(undef,nc)
  sKeₛ = Vector{Array{Float64}}(undef,nc)
  sLVeₛ = Vector{Array{Float64}}(undef,nc)
  sLeₛ = Vector{Array{Float64}}(undef,nc)
  sFeₛ = Vector{Matrix{Float64}}(undef,nc)
  nds_fineₛ = Vector{AbstractVector{Float64}}(undef,nc)
  elem_fineₛ = Vector{Matrix{Int64}}(undef,nc)
  assem_H¹H¹ₛ = Vector{Tuple{Array{Int64}, Array{Int64}, Array{Int64}}}(undef,nc)
  assem_H¹L²ₛ = Vector{Tuple{Array{Int64}, Array{Int64}, Array{Int64}}}(undef,nc)
  patch_elemₛ = Vector{Matrix{Int64}}(undef,nc)
  nds_patchₛ = Vector{AbstractVector{Float64}}(undef,nc)
  patch_indices_to_global_indices = Vector{AbstractVector{Int64}}(undef,nc)
  ms_elem = Vector{Vector{Int64}}(undef,nc)
  global_to_patch_nds = Vector{AbstractVector{Int64}}(undef,nc)

  # Multiscale matrices
  sKms = Vector{Matrix{Float64}}(undef,nc)
  sFms = Vector{Vector{Float64}}(undef,nc)

  for i=1:nc
    start = max(1,i-l)
    last = min(nc,i+l)
    npatch = (last-start+1)
    patch_elemₛ[i] = [elem_conn(i,j) for i=1:npatch, j=0:1] # Check if this needs to stored as well    
    nds_patchₛ[i] = LinRange(first(view(nds_coarse, view(elem_coarse,start,1))), 
                        first(view(nds_coarse, view(elem_coarse,last,2))), npatch+1)
    nfᵢ = aspect_ratio * npatch
    basis_vec_patch[i] = zeros(Float64, nf*q+1, p+1)   

    assem_H¹H¹ₛ[i] = ([H¹Conn(q,i,j) for _=0:q, j=0:q, i=1:nfᵢ],
    [H¹Conn(q,i,j) for j=0:q, _=0:q, i=1:nfᵢ], 
    [H¹Conn(q,i,j) for j=0:q, i=1:nfᵢ])

    assem_H¹L²ₛ[i] = ([H¹Conn(q,i,j) for j=0:q, _=0:p, _=1:npatch, i=1:nfᵢ],
    [L²Conn(p,i,j) for _=0:q, j=0:p, i=1:npatch, _=1:nfᵢ],
    [L²Conn(p,i,j) for j=0:p, i=1:npatch])

    sKeₛ[i] = zeros(Float64, q+1, q+1, nfᵢ)
    sLVeₛ[i] = zeros(Float64, q+1, nfᵢ)
    sLeₛ[i] = zeros(Float64, q+1, p+1, npatch, nfᵢ)
    sFeₛ[i] = zeros(Float64, p+1, npatch)

    sKms[i] = zeros(Float64, npatch*(p+1), npatch*(p+1))
    sFms[i] = zeros(Float64, npatch*(p+1))

    nds_fineₛ[i] = LinRange(nds_patchₛ[i][1], nds_patchₛ[i][end], nfᵢ+1)
    elem_fineₛ[i] = [elem_conn(i,j) for i=1:nfᵢ, j=0:1]
    patch_indices_to_global_indices[i] = (1:length(nds_fineₛ[i])) .+ ((start-1)*aspect_ratio) 
    ms_elem[i] = start*(p+1)-p:last*(p+1)

    global_to_patch_nds[i] = i*(aspect_ratio) - aspect_ratio + 1 : i*(aspect_ratio)+1
  end
  L = zeros(Float64,length(global_to_patch_nds[1]))
  Lᵀ = similar(L)
  ipcache = similar(L)
  fill!(Lᵀ,0.0)
  fill!(ipcache,0.0)
  (nds_coarse, elem_coarse, nds_fine, elem_fine, assem_H¹H¹), 
  (nds_fineₛ, elem_fineₛ), (nds_patchₛ, patch_elemₛ, patch_indices_to_global_indices, global_to_patch_nds, L, Lᵀ, ipcache), 
  basis_vec_patch, 
  (sKeₛ, sLeₛ, sFeₛ, sLVeₛ), (assem_H¹H¹ₛ, assem_H¹L²ₛ, ms_elem), (sKms, sFms)
end

function get_local_basis!(cache, local_basis_vecs::Vector{Matrix{Float64}}, 
  el::Int64, fn::AbstractVector{Int64}, localInd::Int64)
  @assert length(cache) == length(fn)
  lbv = local_basis_vecs[el] 
  copyto!(cache,view(lbv,fn,localInd))
end

function set_local_basis!(local_basis_vecs::Vector{Matrix{Float64}}, 
  el::Int64, fn::AbstractVector{Int64}, localInd::Int64, cache)
  @assert length(cache) == length(fn)
  lbv = local_basis_vecs[el]
  lbv_loc = view(lbv,:,localInd)
  for i=1:lastindex(fn)
    lbv_loc[fn[i]] = cache[i]
  end
end

function split_stiffness_matrix(Kϵ::Array{Float64}, assem_H¹H¹::Tuple{Array{Int64}, Array{Int64}}, global_to_patch_indices::Vector{AbstractVector{Int64}})
  nc = length(global_to_patch_indices)
  coarse_to_fine_stima = Vector{SparseMatrixCSC{Float64,Int64}}(undef,nc)
  iM, jM = assem_H¹H¹
  for t=1:nc
    global_to_elem_indices = global_to_patch_indices[t][1]:global_to_patch_indices[t][end]-1
    iM_elem = @views iM[:,:,global_to_elem_indices] .- (minimum(global_to_elem_indices)-1)
    jM_elem = @views jM[:,:,global_to_elem_indices] .- (minimum(global_to_elem_indices)-1)
    coarse_to_fine_stima[t] = @views sparse(vec(iM_elem), vec(jM_elem), vec(Kϵ[:, :, global_to_elem_indices]))
  end
  coarse_to_fine_stima
end

function split_load_vector(Fϵ::Matrix{Float64}, assem_H¹H¹::Matrix{Int64}, global_to_patch_indices::Vector{AbstractVector{Int64}})
  nc = length(global_to_patch_indices)
  coarse_to_fine_load_vec = Vector{Vector{Float64}}(undef,nc)
  vM = assem_H¹H¹
  for t=1:nc
    global_to_elem_indices = global_to_patch_indices[t][1]:global_to_patch_indices[t][end]-1
    vM_elem = @views vM[:,global_to_elem_indices] .- (minimum(global_to_elem_indices)-1)
    coarse_to_fine_load_vec[t] = @views collect(sparsevec(vec(vM_elem), vec(Fϵ[:,global_to_elem_indices])))
  end
  coarse_to_fine_load_vec
end

function split_stiffness_matrix(Kϵ::SparseMatrixCSC, 
  global_to_patch_indices::Vector{AbstractVector{Int64}})
  nc = length(global_to_patch_indices)
  coarse_to_fine_stima = Vector{SparseMatrixCSC{Float64,Int64}}(undef,nc)
  for t=1:nc
    fine_stima = Kϵ[global_to_patch_indices[t], global_to_patch_indices[t]]
    (t!=1) && (fine_stima[1,1] /=2.0)
    (t!=nc) && (fine_stima[end,end] /=2.0)
    coarse_to_fine_stima[t] = fine_stima
  end
  coarse_to_fine_stima
end

function split_load_vector(Fϵ::AbstractVector{Float64},
  global_to_patch_indices::Vector{AbstractVector{Int64}})
  nc = length(global_to_patch_indices)
  coarse_to_fine_load_vec = Vector{Vector{Float64}}(undef,nc)
  for t=1:nc
    F = Fϵ[global_to_patch_indices[t]]
    (t!=1) && (F[1]/=2.0)
    (t!=nc) && (F[end] /=2.0)
    coarse_to_fine_load_vec[t] = F
  end
  coarse_to_fine_load_vec
end

function mat_contribs!(cache, D::Function)
  full_cache, assem_data, global_to_patch_indices, matcontribs, _ = cache
  elem_cache, bc, quad_data, J_elem, matdata, vecdata, q, quad, index = assem_data
  xqs_elem, Dxqs_elem = quad_data
  xqs, _ = full_cache[3]
  J = full_cache[4]
  (iM_elem, jM_elem, vM_elem), (iV_elem, vV_elem) = matdata, vecdata
  nc = size(global_to_patch_indices,1)
  nf = size(xqs,1)
  aspect_ratio = Int(nf/nc)
  for t=1:nc
    gtpi = global_to_patch_indices[t]
    gtpi1 = view(gtpi,1:aspect_ratio)
    fill!(iM_elem,0); fill!(jM_elem,0); fill!(vM_elem,0.0)
    fill!(iV_elem,0); fill!(vV_elem,0.0)    
    fill!(Dxqs_elem,0.0)
    copyto!(xqs_elem, view(xqs,gtpi1,:))
    copyto!(J_elem, view(J,gtpi1))
    quad_data = xqs_elem, Dxqs_elem
    elem_patch = elem_cache[2]
    matdata = iM_elem, jM_elem, vM_elem
    vecdata = iV_elem, vV_elem    
    assem_elem_cache = (elem_patch, ~), bc, quad_data, J_elem, matdata, vecdata, q, quad, index
    fillsKe!(assem_elem_cache, D)      
    matdata = assem_elem_cache[5] 
    matcontribs[t] = sparse(matdata[1], matdata[2], matdata[3])
  end
  matcontribs
end

function vec_contribs!(cache, f::Function)
  full_cache, assem_data, global_to_patch_indices, _, veccontribs = cache
  elem_cache, bc, quad_data, J_elem, matdata, vecdata, q, quad, index = assem_data
  xqs_elem, Dxqs_elem = quad_data
  xqs, _ = full_cache[3]
  J = full_cache[4]
  (iM_elem, jM_elem, vM_elem), (iV_elem, vV_elem) = matdata, vecdata
  nc = size(global_to_patch_indices,1)
  nf = size(xqs,1)
  aspect_ratio = Int(nf/nc)
  for t=1:nc
    gtpi = global_to_patch_indices[t]
    gtpi1 = view(gtpi,1:aspect_ratio)
    fill!(iM_elem,0); fill!(jM_elem,0); fill!(vM_elem,0.0)
    fill!(iV_elem,0); fill!(vV_elem,0.0)    
    fill!(Dxqs_elem,0.0)
    copyto!(xqs_elem, view(xqs,gtpi1,:))
    copyto!(J_elem, view(J,gtpi1))
    quad_data = xqs_elem, Dxqs_elem
    elem_patch = elem_cache[2]
    matdata = iM_elem, jM_elem, vM_elem
    vecdata = iV_elem, vV_elem    
    assem_elem_cache = (elem_patch, ~), bc, quad_data, J_elem, matdata, vecdata, q, quad, index
    fillsFe!(assem_elem_cache, f)      
    vecdata = assem_elem_cache[6] 
    veccontribs[t] = sparsevec(vecdata[1], vecdata[2])
  end
  veccontribs
end

function mat_vec_contribs_cache(nds::AbstractVector{Float64}, elem::Matrix{Int64}, q::Int64, quad::Tuple{Vector{Float64},Vector{Float64}},
  global_to_patch_indices::Vector{AbstractVector{Int64}})
  nc = size(global_to_patch_indices,1)
  assem_cache = assembler_cache(nds, elem, quad, q)
  _, _, quad_data, J, _, _, q, _, index = assem_cache
  xqs, _ = quad_data
  gtpi = global_to_patch_indices[1]
  npatch = size(gtpi,1)-1 # No of elems in patch
  elem_patch = elem[1:npatch,:]
  nds_patch = nds[1:npatch+1]
  xqs_elem = similar(xqs[1:npatch,:])
  fill!(xqs_elem,0.0)
  Dxqs_elem = similar(xqs_elem)
  fill!(Dxqs_elem,0.0)
  J_elem = J[1:npatch]
  iM_elem = Vector{Int64}(undef, (q+1)^2*npatch)
  jM_elem = Vector{Int64}(undef, (q+1)^2*npatch)
  vM_elem = Vector{Float64}(undef, (q+1)^2*npatch)
  iV_elem = Vector{Int64}(undef, (q+1)*npatch)
  vV_elem = Vector{Float64}(undef, (q+1)*npatch)
  fill!(iM_elem,0); fill!(jM_elem,0); fill!(vM_elem,0.0)
  fill!(iV_elem,0); fill!(vV_elem,0.0)  
  matcontribs = Vector{AbstractMatrix{Float64}}(undef,nc)
  veccontribs = Vector{Vector{Float64}}(undef,nc)
  for t=1:nc
    matcontribs[t] = spzeros(Float64,npatch+1,npatch+1)
    veccontribs[t] = zeros(Float64,npatch+1)
  end

  assem_cache, ((nds_patch, elem_patch), bc, (xqs_elem, Dxqs_elem), J_elem, (iM_elem, jM_elem, vM_elem), (iV_elem, vV_elem), q, quad, index), global_to_patch_indices, matcontribs, veccontribs
end