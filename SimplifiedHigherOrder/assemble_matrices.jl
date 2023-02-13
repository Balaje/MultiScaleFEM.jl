function fillsKe!(sKe::AbstractArray{Float64}, cache, nds_patch::AbstractVector{Float64},
  elem_fine::AbstractVecOrMat{Int64}, q::Int64, quad::Tuple{Vector{Float64}, Vector{Float64}},
  D::Function)
  
  fill!(sKe,0.0)
  nf = size(elem_fine,1)
  for t=1:nf
    cs = view(nds_patch, view(elem_fine, t, :))
    hlocal = cs[2]-cs[1]
    qs,ws = quad
    for i=1:lastindex(qs)
      x̂ = (cs[2]+cs[1])*0.5 + (cs[2]-cs[1])*0.5*qs[i]
      ∇ϕᵢ!(cache, qs[i])
      basis = cache[3]
      for j=1:q+1, k=1:q+1
        sKe[j,k,t] += ws[i]*D(x̂)*basis[j]*basis[k]*(hlocal*0.5)^-1
      end
    end
  end
end

function fillsFe!(sFe::AbstractArray{Float64}, cache, nds_patch::AbstractVector{Float64}, 
  elem_fine::AbstractVecOrMat{Int64}, q::Int64, quad::Tuple{Vector{Float64}, Vector{Float64}}, f::Function) 
  
  fill!(sFe,0.0)
  nf = size(elem_fine,1)
  for t=1:nf
    cs = view(nds_patch, view(elem_fine, t, :))
    hlocal = cs[2]-cs[1]
    qs,ws = quad
    for i=1:lastindex(qs)
      x̂ = (cs[2]+cs[1])*0.5 + (cs[2]-cs[1])*0.5*qs[i]
      Λₖ!(cache, x̂, cs, p)
      basis = cache
      for j=1:q+1
        sFe[j,t] += ws[i]*f(x̂)*basis[j]*(hlocal*0.5)
      end 
    end 
  end
end

function fillsLe!(sLe::AbstractArray{Float64}, basis_cache, nds_fine::AbstractVector{Float64}, nds_coarse::AbstractVector{Float64},
  elem_fine::AbstractVecOrMat{Int64}, elem_coarse::AbstractVecOrMat{Int64}, fespace::Tuple{Int64,Int64}, 
  quad::Tuple{Vector{Float64}, Vector{Float64}})
  
  q,p = fespace
  cache1, cache2 = basis_cache
  fill!(sLe,0.0)
  nf = size(elem_fine,1)
  nc = size(elem_coarse,1)
  qs,ws = quad
  for i=1:lastindex(qs)
    for tf=1:nf
      cq = view(nds_fine, view(elem_fine, tf, :))
      x̂ = (cq[1]+cq[2])*0.5 + (cq[2]-cq[1])*0.5*qs[i]
      ϕᵢ!(cache1, qs[i])
      for tc=1:nc
        cp = view(nds_coarse, view(elem_coarse, tc, :))      
        Λₖ!(cache2, x̂, cp, p)
        basis1 = cache1[3]
        basis2 = cache2
        for k=1:p+1, j=1:q+1
          sLe[j,k,tc,tf] += ws[i]*basis1[j]*basis2[k]*(cq[2]-cq[1])*0.5
        end
      end
    end
  end
end

function fillsKms!(sKms::Vector{Matrix{Float64}}, cache, nc::Int64, p::Int64, l::Int64)  
  local_basis_vecs, global_to_patch_indices, L, Lᵀ, matrix_cache, ipcache = cache  
  for t=1:nc
    start = max(1,t-l)
    last = min(nc,t+l)
    binds = start:last 
    nd = (last-start+1)*(p+1)    
    gtpi = global_to_patch_indices[t]
    for ii=1:nd, jj=1:nd
      fill!(ipcache,0.0)
      ii1 = ceil(Int,ii/(p+1)); ii2 = ceil(Int,jj/(p+1))
      ll1 = ceil(Int,(ii-1)%(p+1)) + 1; ll2 = ceil(Int,(jj-1)%(p+1)) + 1 
      get_local_basis!(L, local_basis_vecs, binds[ii2], gtpi, ll2)     
      get_local_basis!(Lᵀ, local_basis_vecs, binds[ii1], gtpi, ll1) 
      Kₛ = matrix_cache[t]
      mul!(ipcache, Kₛ, Lᵀ)
      for tt=1:lastindex(ipcache)
        sKms[t][ii,jj] += L[tt]*ipcache[tt]
      end   
    end    
  end
end

function fillsFms!(sFms::Vector{Vector{Float64}}, cache, nc::Int64, p::Int64, l::Int64)
  local_basis_vecs, global_to_patch_indices, Lᵀ, vector_cache = cache  
  for t=1:nc
    start = max(1,t-l)
    last = min(nc,t+l)
    binds = start:last     
    nd = (last-start+1)*(p+1)    
    gtpi = global_to_patch_indices[t] 
    for ii=1:nd
      ii1 = ceil(Int,ii/(p+1));
      ll1 = ceil(Int,(ii-1)%(p+1)) + 1;      
      get_local_basis!(Lᵀ, local_basis_vecs, binds[ii1], gtpi, ll1)
      F = vector_cache[t]
      (t==1) 
      for tt=1:lastindex(Lᵀ)
        sFms[t][ii] += F[tt]*Lᵀ[tt]
      end
    end
  end
end

function fillLoadVec!(sFe::AbstractMatrix{Float64}, cache, nds::AbstractVector{Float64}, 
  elem::AbstractVecOrMat{Int64}, 
  q::Int64, quad::Tuple{Vector{Float64}, Vector{Float64}}, f::Function)
  fill!(sFe,0.0)
  nf = size(elem,1)
  for t=1:nf
    cs = view(nds, view(elem, t, :))
    hlocal = cs[2]-cs[1]
    qs,ws = quad
    for i=1:lastindex(qs)
      x̂ = (cs[2]+cs[1])*0.5 + (cs[2]-cs[1])*0.5*qs[i]
      ϕᵢ!(cache, qs[i])
      basis = cache[3]
      for j=1:q+1
        sFe[j,t] += ws[i]*f(x̂)*basis[j]*(hlocal*0.5)
      end 
    end 
  end
end

function assemble_MS!(cache, sKe::Vector{Matrix{Float64}}, sFe::Vector{Vector{Float64}}, ms_elem::Vector{Vector{Int64}})
  nc = size(ms_elem,1)
  Kₘₛ, Fₘₛ = cache
  fill!(Kₘₛ,0.0)
  fill!(Fₘₛ,0.0)
  for t=1:nc
    local_dof = size(ms_elem[t],1)
    elem = ms_elem[t]
    local_mat = sKe[t]
    local_vec = sFe[t]
    for ti=1:local_dof
      Fₘₛ[elem[ti]] += local_vec[ti]
      for tj=1:local_dof
        Kₘₛ[elem[ti],elem[tj]] += local_mat[ti,tj]
      end
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
    for tt=1:lastindex(res)
      res[tt] += sol[(p+1)*j+i-p]*sol_cache[tt]
    end
  end
end


#=
New assemblers 
=#
function fillsKe!(cache, D::Function)
  elemdata, bc, quad_data, J, matdata, _, q, quad, index = cache
  ii,jj,sA = matdata
  fill!(ii,0); fill!(jj,0); fill!(sA,0.0)
  elem = elemdata[1]
  qs, ws = quad
  nc = size(elem,1)
  xqs, Dxqs = quad_data
  map!(D, Dxqs, xqs)
  for p=1:lastindex(qs)
    x̂ = qs[p]
    w = ws[p]
    ∇ϕᵢ!(bc, x̂)
    index = 0
    bases = bc[3]
    for i=1:q+1, j=1:q+1
      setindex!(ii, view(elem,:,i), index+1:index+nc)
      setindex!(jj, view(elem,:,j), index+1:index+nc)      
      @views sA[index+1:index+nc] += Dxqs[:,p].*J.^(-1)*bases[i]*bases[j]*w
      index = index+nc
    end
  end
end

function fillsFe!(cache, g::Function)
  elemdata, bc, quad_data, J, _, vecdata, q, quad, index = cache
  ii,sF = vecdata
  fill!(ii,0); fill!(sF,0.0)
  elem = elemdata[1]
  qs, ws = quad
  nc = size(elem,1)
  xqs, gxqs = quad_data
  map!(g, gxqs, xqs)
  for p=1:lastindex(qs)
    x̂ = qs[p]
    w = ws[p]
    ϕᵢ!(bc, x̂)
    index = 0
    bases = bc[3]
    for i=1:q+1
      setindex!(ii, view(elem,:,i), index+1:index+nc)
      @views sF[index+1:index+nc] += w*bases[i]*(gxqs[:,p].*J)
      index = index+nc
    end
  end
end

function assembler_cache(nds::AbstractVector{Float64}, elem::Matrix{Int64}, 
  quad::Tuple{Vector{Float64},Vector{Float64}}, q::Int64)
  qs, _ = quad
  nds_elem = nds[elem]
  nc = size(elem,1)
  iiM = Vector{Int64}(undef,(q+1)^2*nc)  
  jjM = Vector{Int64}(undef,(q+1)^2*nc)
  sA = Vector{Float64}(undef,(q+1)^2*nc)
  fill!(iiM,0); fill!(jjM,0); fill!(sA,0.0)
  iiV = Vector{Int64}(undef,(q+1)*nc)
  sF = Vector{Float64}(undef,(q+1)*nc)
  fill!(iiV,0); fill!(sF,0.0)
  xqs = Matrix{Float64}(undef, nc, length(qs))
  J = (nds_elem[:,2]-nds_elem[:,1])*0.5
  for i=1:lastindex(qs)
    xqs[:,i] = (nds_elem[:,2]+nds_elem[:,1])*0.5 + (nds_elem[:,2]-nds_elem[:,1])*0.5*qs[i]
  end
  Dxqs = similar(xqs)
  fill!(Dxqs,0.0)
  bc = basis_cache(q)
  index = 0
  (elem,nds), bc, (xqs,Dxqs), J, (iiM,jjM,sA), (iiV,sF), q, quad, index
end