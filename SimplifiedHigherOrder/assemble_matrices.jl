function fillsKe!(sKe::AbstractArray{Float64}, cache, nds_patch::AbstractVector{Float64},
  elem_fine::AbstractVecOrMat{Int64}, q::Int64, quad::Tuple{Vector{Float64}, Vector{Float64}})
  
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
  local_basis_vecs, K, global_to_patch_indices, ipcache = cache  
  for t=1:nc
    start = max(1,t-l)
    last = min(nc,t+l)
    binds = start:last 
    nd = (last-start+1)*(p+1)    
    for ii=1:nd, jj=1:nd
      ii1 = ceil(Int,ii/(p+1)); ii2 = ceil(Int,jj/(p+1))
      ll1 = ceil(Int,(ii-1)%(p+1)) + 1; ll2 = ceil(Int,(jj-1)%(p+1)) + 1
      L = local_basis_vecs[binds[ii2]][global_to_patch_indices[t], ll2]     
      Lᵀ = local_basis_vecs[binds[ii1]][global_to_patch_indices[t], ll1]
      Kₛ = K[global_to_patch_indices[t], global_to_patch_indices[t]]
      (t!=1) && (Kₛ[1,1] /=2.0)
      (t!=nc) && (Kₛ[end,end] /=2.0)
      sKms[t][ii,jj] = L⋅(Kₛ*Lᵀ)
      (t!=1) && (Kₛ[1,1] *=2.0)
      (t!=nc) && (Kₛ[end,end] *=2.0)
    end    
  end
end

function fillsFms!(sFms::Vector{Vector{Float64}}, cache, nc::Int64, p::Int64, l::Int64)
  local_basis_vecs, Fϵₛ, global_to_patch_indices, _ = cache  
  for t=1:nc
    start = max(1,t-l)
    last = min(nc,t+l)
    binds = start:last     
    nd = (last-start+1)*(p+1)     
    for ii=1:nd
      ii1 = ceil(Int,ii/(p+1));
      ll1 = ceil(Int,(ii-1)%(p+1)) + 1;
      F = Fϵₛ[global_to_patch_indices[t]]
      (t!=1) && (F[1]/=2.0)
      (t!=nc) && (F[end] /=2.0)
      Lᵀ = local_basis_vecs[binds[ii1]][global_to_patch_indices[t],ll1]        
      sFms[t][ii] = dot(F,Lᵀ)
      (t!=1) && (F[1]*=2.0)
      (t!=nc) && (F[end] *=2.0)
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