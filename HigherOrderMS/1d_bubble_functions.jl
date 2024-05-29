#### #### #### #### #### #### #### #### #### #### 
# Code to test the bubble function implementation 
#### #### #### #### #### #### #### #### #### #### 

include("HigherOrderMS.jl");

domain = (0.0,1.0)
nc = 16
p = 0
nf = 2^15
l = 4
q = 1;
nds_fine = LinRange(domain..., q*nf+1)
C = _c(domain, nc, p)
elem_coarse = [i+j for i=1:nc, j=0:1]
nds_coarse = LinRange(domain..., nc+1)

plt1 = Plots.plot();
plt2 = Plots.plot()
plt3 = Plots.plot();
plt4 = Plots.plot();

for t = [1,4]
  tri = Tuple(nds_coarse[elem_coarse[t,:]])
  start = max(1,t-1)
  last = min(nc,t+1)    
  if(t==1 || t==nc) 
    patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
  else
    patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[t,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
  end  
  P = (patch[1][1], patch[end][2]); 

  Plots.plot!(plt1, nds_fine, Λₖ!.(nds_fine, Ref(tri), Ref(p), Ref(1)), label="\$ \\Lambda_{1,K} \$", title="Legendre Polynomials")
  Plots.plot!(plt2, nds_fine, bⱼ.(nds_fine, Ref(tri), Ref(C[end]), 1), label="\$ b_{1,K} \$", title="Bubble Functions")
  Plots.plot!(plt3, nds_fine, νⱼ.(nds_fine, t, Ref(C)), label="\$ \\nu_{K} \$", lw=1)
  Plots.plot!(plt3, nds_fine, ιⱼ.(nds_fine, Ref(tri), Ref(P)), label="\$ \\iota_{K} \$ ", lw=1, ls=:dash, title="Auxiliary functions")
  Plots.plot!(plt4, nds_fine, Pₕbⱼ.(nds_fine, t, Ref(C), 1, 1), label="\$ \\iota_{K} + \\nu_{K} \$", title="Extended bubble functions")
end

using Test 
using LinearAlgebra

@testset "Check the L²-projection of the corrected bubble functions" begin
  for p=[1,2,3]    
    C = _c(domain, nc, p)    
    n = ceil(Int64, 0.5*(2*(2p+2)+1));
    x̂, w = gausslegendre(n);        
    Π = zeros(p+1, p+1);   
    for t=1:nc
      tri = Tuple(nds_coarse[elem_coarse[t,:]])
      xqs = (tri[2]+tri[1])/2 .+ (tri[2]-tri[1])/2*x̂  
      for i=1:p+1
        Π[i,i] = sum(w .* Λₖ!.(xqs, Ref(tri), p, i) .* Λₖ!.(xqs, Ref(tri), p, i))*(tri[2]-tri[1])*0.5        
      end

      # Test whether the Legendre polynomials are orthonormal      
      @test Π ≈ I(p+1) 

      # Get the patch
      start = max(1,t-1)
      last = min(nc,t+1)    
      if(t==1 || t==nc) 
        patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
      else
        patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[t,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
      end        
      P = (patch[1][1], patch[end][2]); 

      # Compute the L² projection of the zero-th order bubble functions
      F1 = zeros(p+1);        
      for i=1:p+1
        F1[i] = sum(w .* bⱼ.(xqs, Ref(tri), Ref(C[end]), 1) .* Λₖ!.(xqs, Ref(tri), p, i))*(tri[2]-tri[1])*0.5                
      end

      # Compute the L² projection of the zero-th order extended bubble functions
      F2 = zeros(p+1);        
      for i=1:p+1
        F2[i] = sum(w .* Pₕbⱼ.(xqs, t, Ref(C), 1, 1) .* Λₖ!.(xqs, Ref(tri), p, i))*(tri[2]-tri[1])*0.5                
      end

      function E1(p)
        res = zeros(p+1)
        res[1] = 1.0
        res
      end

      # Test if the L² projection of the extended bubble is equal to the Legendre polynomial      
      @test F1 ≈ E1(p)
      @test F2 ≈ E1(p)

    end
  end
end; # All tests should pass

##### ##### ##### ##### ##### ##### ##### ##### 
# Compute the zero-th order improved MS basis #
##### ##### ##### ##### ##### ##### ##### #####

"""
Function to assemble the matrix associated with the RHS of the saddle point problem
"""
plot_spe = Plots.plot()
function assemble_lm_bubble_matrix(domain, nc::Int64, p::Int64)  
  C = _c(domain, nc, p) 
  nds = C[3]
  elem = C[2]
  n = ceil(Int64, 0.5*(2*(2p+2)+1));
  x̂, w = gausslegendre(n);
  l2mat = zeros(nc*(p+1), nc*(p+1))
  for t=1:nc           
    tri_t = Tuple(nds[elem[t,:]])
    for k=max(1,t-1):min(nc,t+1)           
    # for k=[t]
      tri = Tuple(nds[elem[k,:]])    
      xqs = (tri[2]+tri[1])/2 .+ (tri[2]-tri[1])/2*x̂  
      # (t==1) && Plots.plot!(plot_spe, nds_fine, Pₕbⱼ.(nds_fine, t, Ref(C), 1, 1), label="")      
      for i=1:p+1, j=1:p+1        
        # (t==1) && Plots.plot!(plot_spe, nds_fine, Λₖ!.(nds_fine, Ref(tri), p, i), label="", ls=:dash, lw=1)
        l2mat[(t-1)*(p+1)+i, (k-1)*(p+1)+j] += sum(w .* (Λₖ!.(xqs, Ref(tri_t), p, i) - Pₕbⱼ.(xqs, t, Ref(C), 1, 0)) .* Λₖ!.(xqs, Ref(tri), p, i))*(tri[2]-tri[1])*0.5                
        # l2mat[(t-1)*(p+1)+i, (k-1)*(p+1)+j] = sum(w .* Λₖ!.(xqs, Ref(tri_t), p, i) .* Λₖ!.(xqs, Ref(tri), p, j))*(tri[2]-tri[1])*0.5                
      end                  
    end       
  end  
  l2mat
end

"""
Function to compute the multiscale basis function using the zero-th order extended bubble function
"""
function compute_ms_basis_bubble(fspace::FineScaleSpace, D::Function, p::Int64, nc::Int64, l::Int64, 
  patch_indices_to_global_indices::Vector{AbstractVector{Int64}}, coarse_indices_to_fine_indices::Vector{AbstractVector{Int64}}, α::Int64, β::Int64)
  nf = fspace.nf
  q = fspace.q
  basis_vec_ms = spzeros(Float64,q*nf+1,(p+1)*nc) # To store the multiscale basis functions
  Λ̃ = assemble_lm_bubble_matrix(fine_scale_space.domain, nc, p)
  K, L, Λ = get_saddle_point_problem(fspace, D, p, nc)    
  C_data = _c(domain, nc, p)
  # _, elem_coarse, nds_coarse, _ = C_data
  nds_fine = LinRange(domain..., q*nf+1)
  index = 1
  for t=1:nc    
    fullnodes = patch_indices_to_global_indices[t]
    bnodes = [fullnodes[1], fullnodes[end]]
    freenodes = setdiff(fullnodes, bnodes)
    start = max(1,t-l)
    last = min(nc,t+l)
    gn = start*(p+1)-p:last*(p+1)    
    stima_el = K[freenodes,freenodes]
    lmat_el = L[freenodes,gn]    

    # j=1 (zero-th order function)      
    # ν(y) = Pₕbⱼ(y[1], t, C_data, 0, 1)
    # ι(y) = Pₕbⱼ(y[1], t, C_data, 1, 0)    
    bₖ(y) = Pₕbⱼ(y[1], t, C_data, 1, 1)    

    # Correcting νₖ    
    # for k=max(1,t-1):min(nc,t+1)     
    for k=max(1,t-1):min(nc,t+1)
      fullnodes₁ = coarse_indices_to_fine_indices[k] 
      bnodes₁ = [fullnodes₁[1], fullnodes₁[end]]   
      freenodes₁ = setdiff(fullnodes₁, bnodes₁)      
      # l-patch of k (Saddle point problem domain)
      fullnodes₂ = patch_indices_to_global_indices[k]
      bnodes₂ = [fullnodes₂[1], fullnodes₂[end]]
      freenodes₂ = setdiff(fullnodes₂, bnodes₂)
      gn₂ = max(1,k-l)*(p+1)-p:min(nc,k+l)*(p+1)      
      # Solve the problem for R(v - Iₕv)      
      # Evaluate LHS on the patch
      lhs = [K[freenodes₂,freenodes₂] L[freenodes₂,gn₂]; L[freenodes₂,gn₂]' spzeros(Float64, length(gn₂), length(gn₂))]                       
      # Evaluate RHS by evaluating νₖ() on the element fine-scale DOFs            
      rhs₁ = [K[freenodes₂,freenodes₁]*bₖ.(nds_fine[freenodes₁]); zeros(length(gn₂))]            
      # rhs₁ = [zeros(length(freenodes₂)); -Λ̃[gn₂,index]]            
      # Solve to obtain the corrector
      basis_vec_ms[freenodes₂,index] += (-(lhs\rhs₁)[1:length(freenodes₂)])                  
    end 

    # Add the extended bubble after adding the correction.
    basis_vec_ms[fullnodes, index] += bₖ.(nds_fine[fullnodes])
    # basis_vec_ms[fullnodes, index] += β*ν.(nds_fine[fullnodes])    
        
    index += 1   

    for _=2:p+1
      fvecs_el = [zeros(length(freenodes)); Λ[gn, index]]
      lhs = [stima_el lmat_el; (lmat_el)' spzeros(Float64, length(gn), length(gn))]
      rhs = fvecs_el           
      sol = lhs\rhs                 
      basis_vec_ms[freenodes,index] = sol[1:length(freenodes)]
      index += 1   
    end
  end
  basis_vec_ms
end

# Test out the method

# Get the Gridap fine-scale description
fine_scale_space = FineScaleSpace(domain, q, 4, nf);

# Compute the map between the coarse and fine scale
patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));

# Compute Multiscale bases with some Diffusion coefficient
# D(x) = (0.25 + 0.125*cos(2π*x[1]/2^-5))^-1;
plt5 = Plots.plot();

D(x) = 1.0;
els = [4]
# D(x) = (2 + cos(2π*x[1]/2^-5))^-1;
basis_vec_ms₁ = compute_ms_basis(fine_scale_space, D, p, nc, l, patch_indices_to_global_indices);
for el=els
  i = el*(p+1)-p
  Plots.plot!(plt5, nds_fine, basis_vec_ms₁[:,i], ls=:dash, lw=1, label="Old basis function "*string(i), lc=:blue)
end
for αβ=[(1,0)]
  local α, β = αβ
  local basis_vec_ms₂ = compute_ms_basis_bubble(fine_scale_space, D, p, nc, l, patch_indices_to_global_indices, coarse_indices_to_fine_indices, α, β);
  for el=els
    i = el*(p+1)-p
    @show i
    if(α==1 && β==1)
      Plots.plot!(plt5, nds_fine, basis_vec_ms₂[:,i], lw=0.5, label="\$ (1-C^l)(\\nu_k+\\iota_k) \$", lc=:green)
    elseif(α==0)
      Plots.plot!(plt5, nds_fine, basis_vec_ms₂[:,i], lw=0.5, label="\$ (1-C^l)(\\nu_k) \$", lc=:magenta)
    else
      Plots.plot!(plt5, nds_fine, basis_vec_ms₂[:,i], lw=0.5, label="\$ (1-C^l)(\\iota_k) \$", lc=:red)      
    end
  end
end
