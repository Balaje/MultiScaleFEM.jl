module MultiscaleBases

using ProgressMeter
using SparseArrays

using HigherOrderMS_1d.CoarseBases: Λₖ!, ιₖ, _c
using HigherOrderMS_1d.CoarseToFine: coarse_space_to_fine_space
using HigherOrderMS_1d.Assemblers: FineScaleSpace, get_saddle_point_problem

#### ##### ##### ##### ##### ##### ##### ##### 
# Module containing the functions to obtain: #           
# 1) the multiscale bases.                   #
# 2) the stabilized multiscale bases         #
#### ##### ##### ##### ##### ##### ##### #####
function compute_ms_basis(fspace::FineScaleSpace, D::Function, p::Int64, nc::Int64, l::Int64, 
                          patch_indices_to_global_indices::Vector{AbstractVector{Int64}}; T=Float64)
  nf = fspace.nf
  q = fspace.q
  basis_vec_ms = spzeros(T,q*nf+1,(p+1)*nc) # To store the multiscale basis functions
  K, L, Λ = get_saddle_point_problem(fspace, D, p, nc)
  f1 = zeros(T,size(K,1))
  index = 1
  @showprogress desc="Computing basis functions" for t=1:nc
    fullnodes = patch_indices_to_global_indices[t]
    bnodes = [fullnodes[1], fullnodes[end]]
    bvals = [0.0,0.0]
    freenodes = setdiff(fullnodes, bnodes)
    start = max(1,t-l)
    last = min(nc,t+l)
    gn = start*(p+1)-p:last*(p+1)    
    stima_el = K[freenodes,freenodes]
    lmat_el = L[freenodes,gn]
    loadvec_el = (f1 - K[:,bnodes]*bvals)
    for _=1:p+1
      fvecs_el = [loadvec_el[freenodes]; Λ[gn, index]]
      lhs = [stima_el lmat_el; (lmat_el)' spzeros(T, length(gn), length(gn))]
      rhs = fvecs_el           
      sol = lhs\rhs                 
      basis_vec_ms[freenodes,index] = sol[1:length(freenodes)]
      index += 1   
    end
  end
  basis_vec_ms
end

"""
Compute the correction of ιₖ(x) + νₖ(x)
"""
function compute_stabilized_ms_basis(fspace::FineScaleSpace, D::Function, p::Int64, nc::Int64, l::Int64; T=Float64)
  domain = fspace.domain
  nf = fspace.nf
  q = fspace.q
  basis_vec_ms = spzeros(T,q*nf+1,nc) # To store the stabilized basis functions (to return)  
  nds_fine = LinRange(domain...,q*nf+1)
  K, L, _ = get_saddle_point_problem(fspace, D, p, nc; T=T)  
  # We need this to obtain the 1-patch and the element for the ιₖ component
  elem_coarse = [i+j for i=1:nc, j=0:1]
  nds_coarse = LinRange(domain..., nc+1) 
  elem_indices_to_global_indices = coarse_space_to_fine_space(nc, nf, 0, (1,p))[1]; 
  patch_indices_to_global_indices = coarse_space_to_fine_space(nc, nf, l, (1,p))[1];
  # Compute the old multiscale bases for the νₖ component
  β = compute_ms_basis(fspace, D, p, nc, l, patch_indices_to_global_indices; T=T)

  @showprogress desc="Computing Stabilization" for t=1:nc 
    start = max(1,t-1); last = min(nc, t+1); # N¹(G)
    # Get the N¹(K) patch
    # start₁ = max(1,t-1); last₁ = min(nc, t+1); # N¹(G)
    if(t==1 || t==nc) 
      P = Tuple(nds_coarse[elem_coarse[start,:]]), 
          Tuple(nds_coarse[elem_coarse[last,:]])
    else
      P = Tuple(nds_coarse[elem_coarse[start,:]]), 
          Tuple(nds_coarse[elem_coarse[t,:]]), 
          Tuple(nds_coarse[elem_coarse[last,:]])
    end

    if(t==1)
      inds_1 = [t,t+1]
      inds_2 = [1,2]
    elseif(t==nc)
      inds_1 = [t-1,t]
      inds_2 = [1,2]
    else
      inds_1 = [t-1,t,t,t+1]
      inds_2 = [1,2,3,4]
    end
    for (u,u1)=zip(inds_2,inds_1)      
      # G ∈ {K-1, K, K+1}
      startᵤ = max(1,u1-l); lastᵤ = min(nc, u1+l); # Nˡ(G)               
      
      fullnodes = patch_indices_to_global_indices[u1]
      bnodes = [fullnodes[1], fullnodes[end]]
      freenodes = setdiff(fullnodes, bnodes)
      gn = startᵤ*(p+1)-p:lastᵤ*(p+1)          
      stima_el = K[freenodes,freenodes]
      lmat_el = L[freenodes,gn]

      # Extract the fine-scale node in the element
      loadvec = zeros(T, length(nds_fine)); # To store the RHS
      fullnodes₁ = elem_indices_to_global_indices[u1] 
      bnodes₁ = [fullnodes₁[1], fullnodes₁[end]]
      # Source term
      K[bnodes₁,bnodes₁]/=2            
      iota = ιₖ.(nds_fine, Ref(P), Ref(u); T=T)[fullnodes₁]# ιₖ function on the element
      Kel = K[fullnodes₁, fullnodes₁]             
      loadvec[fullnodes₁] = Kel*iota      
      K[bnodes₁,bnodes₁]*=2  
    
      # Solve the saddle point problem
      lhs = [stima_el lmat_el; (lmat_el)' spzeros(T, length(gn), length(gn))]  
      rhs = [-loadvec[freenodes]; zeros(T, length(gn))]    
      sol = lhs\rhs

      # basis_vec_ms = (1-Cˡₖ)ι
      basis_vec_ms[fullnodes,t] += [0.0; sol[1:length(freenodes)]; 0.0] 
      basis_vec_ms[fullnodes₁[2:end],t] += iota[2:end]            
    end    

    # Coefficients for νₖ
    C = vec(_c(nc, t, p; T=T))
    βi = β[:, start*(p+1)-p:last*(p+1)]   

    # basis_vec_ms += ΣcₖΛ̃ₖ
    sol1 = βi*C
    basis_vec_ms[:,t] += sol1
  end
  basis_vec_ms
end

end
