include("HigherOrderMS.jl");
include("corrected_basis.jl");

"""
Construct the time dependent basis functions
"""
function time_dependent_ms_basis(fine_scale_space::FineScaleSpace, D::Function, 
                                 p::Int64, nc::Int64, l::Int64, 
                                 patch_indices_to_global_indices::Vector{AbstractVector{Int64}}, 
                                 BDF::Int64, tf::Float64, Δt::Float64)

  ntime = ceil(Int64, (tf/Δt))

  basis_vec_ms = [spzeros(Float64, q*nf+1, (p+1)*nc) for i=1:ntime]
  
  K, L, Λ = get_saddle_point_problem(fine_scale_space, D, p, nc)
  M = assemble_mass_matrix(fine_scale_space, x->1.0)  

  nds_fine = LinRange(fine_scale_space.domain..., fine_scale_space.q*fine_scale_space.nf+1)
  
  index = 1
  for coarse_el=1:nc
    fullnodes = patch_indices_to_global_indices[coarse_el]
    bnodes = [fullnodes[1], fullnodes[end]]
    freenodes = setdiff(fullnodes, bnodes)
    start = max(1,coarse_el-l)
    last = min(nc,coarse_el+l)
    gn = start*(p+1)-p:last*(p+1)    
    stima_el = K[freenodes,freenodes]
    massma_el = M[freenodes,freenodes]
    lmat_el = L[freenodes,gn]
    
    # Initial condition for the basis
    function ł(cache, tₙ::Float64)
      freenodes, Λₜ = cache
      [zeros(length(freenodes))*tₙ; Λₜ]
      # zeros(length(freenodes))
    end 
    
    for _=1:p+1      
      stima₁ = [stima_el lmat_el; (lmat_el)' spzeros(Float64, length(gn), length(gn))]
      massma₁ = [massma_el zero(lmat_el); zero(lmat_el') spzeros(Float64, length(gn), length(gn))]
      U₀ = [zeros(Float64, length(freenodes)); zeros(Float64, length(gn))]      
      # stima₁ = stima_el
      # massma₁ = massma_el
      # U₀ = sin.(π*nds_fine[freenodes])
      
      ###### ###### ###### ###### ###### ###### 
      #  Solve the time dependent problem
      ###### ###### ###### ###### ###### ###### 
      fcache = freenodes, Λ[gn, index]
      t = 0.0
      for i=1:BDF-1
        dlcache = get_dl_cache(i)
        cache = dlcache, fcache        
        U₁ = BDFk!(cache, t, U₀, Δt, stima₁, massma₁, ł, i)                
        basis_vec_ms[i][freenodes, index] .=  U₁[1:length(freenodes)]        
        U₀ = hcat(U₁, U₀)        
        t += Δt
      end      
      # Remaining BDF steps
      dlcache = get_dl_cache(BDF)
      cache = dlcache, fcache
      for i=BDF:ntime
        U₁ = BDFk!(cache, t+Δt, U₀, Δt, stima₁, massma₁, ł, BDF)
        basis_vec_ms[i][freenodes, index] .=  U₁[1:length(freenodes)]        
        U₀[:,2:BDF] = U₀[:,1:BDF-1]        
        U₀[:,1] = U₁
        t += Δt
      end
      ###### ###### ###### ###### ###### 
      # End time dependent problem
      ###### ###### ###### ###### ###### 

      index += 1
    end    
  end
  basis_vec_ms
end

domain = (0.0,1.0)

nc = 4;
l = 7;
p = 0;
nf = 2^15;
q = 1;
qorder = 4;
fine_scale_space = FineScaleSpace(domain, q, qorder, nf)
nds_fine = LinRange(domain..., q*nf+1);
A(x) = 1.0
f(x,t) = 0.0
u₀(x) = sin(π*x[1])

# Define the time discretization parameters
tf = 1.0
Δt = 1e-2;
ntime = ceil(Int64, tf/Δt);
BDF = 1;

stima = assemble_stiffness_matrix(fine_scale_space, A);
massma = assemble_mass_matrix(fine_scale_space, x->1.0);
fullnodes = 1:q*nf+1;
bnodes = [1, q*nf+1];
freenodes = setdiff(fullnodes, bnodes);
function fₙϵ!(cache, tₙ::Float64)
  fspace, freenodes = cache
  F = assemble_load_vector(fspace, y->f(y,tₙ))
  F[freenodes]
end
# Time marching
let 
  U₀ = u₀.(nds_fine[freenodes])
  global Uex = zero(U₀)  
  t = 0.0
  # Starting BDF steps (1...k-1) 
  fcache = fine_scale_space, freenodes
  for i=1:BDF-1
    dlcache = get_dl_cache(i)
    cache = dlcache, fcache
    U₁ = BDFk!(cache, t, U₀, Δt, stima[freenodes,freenodes], massma[freenodes,freenodes], fₙϵ!, i)
    U₀ = hcat(U₁, U₀)
    t += Δt
  end
  # Remaining BDF steps
  dlcache = get_dl_cache(BDF)
  cache = dlcache, fcache
  for i=BDF:ntime
    U₁ = BDFk!(cache, t+Δt, U₀, Δt, stima[freenodes,freenodes], massma[freenodes,freenodes], fₙϵ!, BDF)
    U₀[:,2:BDF] = U₀[:,1:BDF-1]
    U₀[:,1] = U₁
    t += Δt
  end
  Uex = U₀[:,1] # Final time solution
end


### Solve using the multiscale method
patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
basis_vec_ms = time_dependent_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices, BDF, tf, Δt);
Uₘₛ = setup_initial_condition(u₀, basis_vec_ms[1], fine_scale_space);
for i=1:ntime-1
  Λₖ = basis_vec_ms[i+1]
  Λₖ₋₁ = basis_vec_ms[i]
  Kₘₛ = Λₖ'*stima*Λₖ; 
  Mₘₛ = Λₖ'*massma*Λₖ;
  LHS = Mₘₛ + Δt*Kₘₛ
  F = assemble_load_vector(fine_scale_space, y->f(y, (i+1)*Δt))
  RHS = (Δt)*Λₖ'*F + (Λₖ'*massma*Λₖ₋₁)*Uₘₛ
  global Uₘₛ = (LHS\RHS)
end