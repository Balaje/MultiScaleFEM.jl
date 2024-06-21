##### ###### ###### ###### ###### ###### ###### ###### #
# Program to implement the corrected basis function
##### ###### ###### ###### ###### ###### ###### ###### #
include("HigherOrderMS.jl");

function compute_corrected_basis_function(fine_scale_space::FineScaleSpace, D::Function, p::Int64, nc::Int64, l::Int64, 
                                          patch_indices_to_global_indices::Vector{AbstractVector{Int64}}, δ::Int64, Δt::Float64)
  ### To build the basis functions
  
end

# Coarse scale space parameters
domain = (0.0,1.0)
nc = 16;
p = 1;
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
A(x; nds_micro = nds_micro, diffusion_micro = diffusion_micro) = _D(x[1], nds_micro, diffusion_micro)
Λ = compute_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices);

Plots.plot!(nds_fine, (abs.(Λ[:,1])), label="Old Basis")