##### ###### ###### ###### ###### ###### ###### ###### #
# Program to implement the corrected basis function
##### ###### ###### ###### ###### ###### ###### ###### #
include("HigherOrderMS.jl");

function compute_l2_orthogonal_basis(fine_scale_space::FineScaleSpace, D::Function, p::Int64, nc::Int64, l::Int64, 
                                     patch_indices_to_global_indices::Vector{AbstractVector{Int64}})                                     
  ### To build the basis functions
  nf = fine_scale_space.nf
  q = fine_scale_space.q
  basis_vec_ms = spzeros(Float64,q*nf+1,(p+1)*nc) # To store the multiscale basis functions
  K, L, Λ = get_saddle_point_problem(fine_scale_space, D, p, nc)
  M = assemble_mass_matrix(fine_scale_space, x->1.0)
  β = compute_ms_basis(fine_scale_space, D, p, nc, l, patch_indices_to_global_indices)  
  index = 1
  for t=1:nc
    fullnodes₁ = patch_indices_to_global_indices[t]    
    bnodes₁ = [fullnodes₁[1], fullnodes₁[end]]        
    freenodes₁ = setdiff(fullnodes₁, bnodes₁)    
    start₁ = max(1,t-l); last₁ = min(nc,t+l)    
    gn₁ = start₁*(p+1)-p:last₁*(p+1)    
    stima_el = K[freenodes₁,freenodes₁]
    lmat_el = L[freenodes₁,gn₁]
    for _=1:p+1
      fvecs_el = [M[freenodes₁,freenodes₁]*β[freenodes₁,index]; zeros(Float64,length(gn₁))]
      lhs = [stima_el lmat_el; (lmat_el)'  spzeros(Float64, length(gn₁), length(gn₁))]
      rhs = fvecs_el           
      sol = lhs\collect(rhs)
      basis_vec_ms[freenodes₁, index] = sol[1:length(freenodes₁)]
      index += 1   
    end
  end
  basis_vec_ms
end

#= # Coarse scale space parameters
domain = (0.0,1.0)
nc = 16;
p = 3;
l = 4; 

# Fine scale space parameters
q = 1;
nf = 2^15;
qorder = 6;
fine_scale_space = FineScaleSpace(domain, q, qorder, nf);

nds_fine = LinRange(domain..., q*nf+1);

patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));

Neps = 2^8
nds_micro = LinRange(domain[1], domain[2], Neps+1)
diffusion_micro = 0.5 .+ 0.5*rand(Neps+1)
function _D(x::Float64, nds_micro::AbstractVector{Float64}, diffusion_micro::Vector{Float64})
  n = size(nds_micro, 1)
  for i=1:n
    if(nds_micro[i] < x < nds_micro[i+1])      
      return diffusion_micro[i+1]
    elseif(x==nds_micro[i])
      return diffusion_micro[i+1]
    elseif(x==nds_micro[i+1])
      return diffusion_micro[i+1]
    else
      continue
    end 
  end
end
# A(x; nds_micro = nds_micro, diffusion_micro = diffusion_micro) = _D(x[1], nds_micro, diffusion_micro)
A(x) = 1.0
Λ = compute_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices)
Λₗ = compute_l2_orthogonal_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices);

nb = 9;
Plots.plot();
Plots.plot!(nds_fine, Λₗ[:,nb], label="New Basis"); =#
# Plots.plot!(nds_fine, Λ[:,nb], label="Old Basis");
