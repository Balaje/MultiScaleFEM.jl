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

function fillsKms!(sKms::AbstractArray{Float64}, cache, 
  nds::AbstractVector{Float64},
  elem_ms::AbstractVecOrMat{Int64}, p::Int64, l::Int64, 
  quad::Tuple{Vector{Float64}, Vector{Float64}}; Nfine=200)

  fill!(sKms,0.0)
  KDTrees, Basis, local_basis_vecs, basis_cache = cache
  elem, _ = basis_cache
  nc = size(elem_ms,1)
  qs, ws = quad  
  
  for i=1:lastindex(qs)
    for t=1:nc
      start = max(1,(t-l)) - (((t+l) > nc) ? abs(t+l-nc) : 0) # Start index of patch
      last = min(nc,(t+l)) + (((t-l) < 1) ? abs(t-l-1) : 0) # Last index of patch
      binds = start:last      
      ndofs = (2l+1)*(p+1)
      k=0
      for pᵢ=1:lastindex(binds), qᵢ=1:p+1
        k+=1
        local_basis_vecs[:,k] = view(Basis,:, view(binds,pᵢ), qᵢ)
      end    
      cs = view(nds, view(elem, t, :))      
      hlocal = (cs[2]-cs[1])/Nfine
      xlocal = cs[1]:hlocal:cs[2] 
      for j=1:lastindex(xlocal)-1, pᵢ=1:ndofs, qᵢ=1:ndofs
        x̂ = (xlocal[j+1]+xlocal[j])*0.5 + (xlocal[j+1]-xlocal[j])*0.5*qs[i]        
        sKms[pᵢ,qᵢ,t] += ws[i]*D(x̂)*
          (∇Λₖ(basis_cache, x̂, view(local_basis_vecs,:,pᵢ), @views KDTrees[t]))*
          (∇Λₖ(basis_cache, x̂, view(local_basis_vecs,:,qᵢ), @views KDTrees[t]))*
          ((xlocal[j+1]-xlocal[j])*0.5)
      end      
    end
  end
end

function fillsFms!(sFms::AbstractArray{Float64}, cache, 
  nds::AbstractVector{Float64},
  elem_ms::AbstractVecOrMat{Int64}, p::Int64, l::Int64, 
  quad::Tuple{Vector{Float64}, Vector{Float64}}, f::Function; Nfine=200)

  fill!(sFms,0.0)
  KDTrees, Basis, local_basis_vecs, basis_cache = cache
  elem, _ = basis_cache
  nc = size(elem_ms,1)
  qs, ws = quad  
  for i=1:lastindex(qs)
    for t=1:nc
      start = max(1,(t-l)) - (((t+l) > nc) ? abs(t+l-nc) : 0) # Start index of patch
      last = min(nc,(t+l)) + (((t-l) < 1) ? abs(t-l-1) : 0) # Last index of patch
      binds = start:last      
      ndofs = (2l+1)*(p+1)
      k=0
      for pᵢ=1:lastindex(binds), qᵢ=1:p+1
        k+=1
        local_basis_vecs[:,k] = view(Basis,:, view(binds,pᵢ), qᵢ)
      end      
      cs = view(nds, view(elem, t, :))      
      hlocal = (cs[2]-cs[1])/Nfine
      xlocal = cs[1]:hlocal:cs[2] 
      for j=1:lastindex(xlocal)-1, pᵢ=1:ndofs
        x̂ = (xlocal[j+1]+xlocal[j])*0.5 + (xlocal[j+1]-xlocal[j])*0.5*qs[i]    
        sFms[pᵢ,t] += ws[i]*f(x̂)*
        (Λₖ(basis_cache, x̂, view(local_basis_vecs,:,pᵢ), @views KDTrees[t]))*
        ((xlocal[j+1]-xlocal[j])*0.5)
      end      
    end  
  end
end