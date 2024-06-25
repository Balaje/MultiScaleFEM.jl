include("HigherOrderMS.jl");
include("corrected_basis.jl");

"""
Construct the time dependent basis functions
"""
function time_dependent_ms_basis(fine_scale_space::FineScaleSpace, D::Function, 
  p::Int64, nc::Int64, l::Int64, 
  patch_indices_to_global_indices::Vector{AbstractVector{Int64}}, 
  BDF::Int64, tf::Float64, Δt::Float64, N::Int64)
  
  ntime = ceil(Int64, (tf/Δt))
  
  basis_vec_ms = [spzeros(Float64, q*nf+1, (p+1)*nc) for i=1:ntime+1]
  
  K, L, Λ = get_saddle_point_problem(fine_scale_space, D, p, nc)
  M = assemble_mass_matrix(fine_scale_space, x->1.0)      

  # nds_fine = LinRange(fine_scale_space.domain..., fine_scale_space.q*fine_scale_space.nf+1)
  
  β = compute_ms_basis(fine_scale_space, D, p, nc, l, patch_indices_to_global_indices)
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
      f, Λ₀ = cache
      [f; Λ₀]
      # zeros(length(freenodes))
    end     
    
    Z_el = spzeros(Float64, length(gn), length(gn))
    for _=1:p+1      
      stima₁ = [stima_el lmat_el; 
                (lmat_el)' Z_el]
      massma₁ = [massma_el zero(lmat_el); 
                zero(lmat_el') Z_el]
      # U₀ = zeros(Float64, length(freenodes)+length(gn))
      U₀ = [collect(β[freenodes,index]); zeros(length(gn))]
      
      ###### ###### ###### ###### ###### ###### 
      #  Solve the time dependent problem
      ###### ###### ###### ###### ###### ###### 
      fcache = zeros(length(freenodes)), zeros(Float64,length(gn))
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
      for i=BDF:N
        U₁ = BDFk!(cache, t+Δt, U₀, Δt, stima₁, massma₁, ł, BDF)
        basis_vec_ms[i][freenodes, index] .=  U₁[1:length(freenodes)]
        U₀[:,2:BDF] = U₀[:,1:BDF-1]        
        U₀[:,1] = U₁
        t += Δt
      end

      for i=N+1:ntime+1
        basis_vec_ms[i] = basis_vec_ms[N]
      end
      ###### ###### ###### ###### ###### 
      # End time dependent problem
      ###### ###### ###### ###### ###### 
      
      index += 1
    end    
  end
  basis_vec_ms
end


"""
Modified BDF-k routine
"""
function BDFk!(cache, tₙ::Float64, U::AbstractVecOrMat{Float64}, Δt::Float64, 
  K::AbstractMatrix{Float64}, M::Vector{T}, f!::Function, k::Int64) where T <: AbstractMatrix{Float64}
  # U should be arranged in descending order (n+k), (n+k-1), ...
  # M should be arranged in descending order (n+k), (n+k-1), ...
  @assert (size(U,2) == k) # Check if it is the right BDF-k
  dl_cache, fcache = cache
  coeffs = dl!(dl_cache, k)
  RHS = 1/coeffs[k+1]*(Δt)*(f!(fcache, tₙ+k*Δt))    
  for i=0:k-1    
    RHS += -(coeffs[k-i]/coeffs[k+1])*M[i+1]*U[:,i+1]
  end 
  LHS = (M[1] + 1.0/(coeffs[k+1])*Δt*K)
  Uₙ₊ₖ = LHS\RHS
  Uₙ₊ₖ
end


#######################
# Test out the method #
#######################

"""
Projection of the source function to the MS space
"""
function fₙ!(cache, tₙ::Float64)
  fspace, basis_vec_ms = cache
  F = assemble_load_vector(fspace, y->f(y, tₙ))
  basis_vec_ms'*F
end

"""
Function to setup the initial condition
"""
function setup_initial_condition(u₀::Function, Λ::NTuple{2,SparseMatrixCSC{Float64,Int64}}, fspace::FineScaleSpace)
  Λᵣ, Λₜ = Λ
  massma = assemble_mass_matrix(fspace, x->1.0)
  loadvec = assemble_load_vector(fspace, u₀)
  Mₘₛ = Λₜ'*massma*Λᵣ
  Lₘₛ = Λₜ'*loadvec
  Mₘₛ\Lₘₛ
end 

domain = (0.0,1.0)
# nc = 4;
# l = 7;
p = 3;
nf = 2^14;
q = 1;
qorder = 6;
fine_scale_space = FineScaleSpace(domain, q, qorder, nf)
nds_fine = LinRange(domain..., q*nf+1);
# Random diffusion coefficient
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
# A(x) = (2 + cos(2π*x[1]/2^0))^-1
f(x,t) = sin(π*x[1])
u₀(x) = 0.0
# Define the time discretization parameters
Δt = 1e-3;
tf = 0.5;
ntime = ceil(Int64, tf/Δt);
BDF = 4;

###################################
#  Compute the reference solution #
###################################
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
Uₕ = TrialFESpace(fine_scale_space.U, 0.0)
uₕ = FEFunction(Uₕ, vcat(0.0,Uex,0.0))

##### ########## ########## ########## ########## ##
# Compute the solution using the multiscale method #
##### ########## ########## ########## ########## ##
N = [1,2,4,8,16]
plt = Plots.plot();
plt1 = Plots.plot();
L²Error = zeros(Float64,size(N));
H¹Error = zeros(Float64,size(N));
for l=[7]
  fill!(L²Error, 0.0)
  fill!(H¹Error, 0.0)
  for (nc,itr) in zip(N, 1:lastindex(N))
    # Compute the time dependent multiscale basis
    local patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));    

    # Compute the original MS method bases.
    global Λₜ = compute_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices)         
    # global Λ = time_dependent_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices, BDF, tf, Δt, min(ntime, 20));       
    global Λ = fill(Λₜ, ntime+1)

    # Compute the projection of the fine scale matrices to the multiscale space    
    𝐊 = [Λₜ'*stima*Λ[i+1] for i=1:ntime]; # Stiffness matrix    
    Mₘₛ¹ = [[[Λₜ'*massma*Λ[i+1]]; [Λₜ'*massma*Λ[i+1-k] for k=1:i]] for i=1:BDF-1]# Compute the vector of mass matrices    
    Mₘₛ² = [[[Λₜ'*massma*Λ[i+1]]; [Λₜ'*massma*Λ[i+1-k] for k=1:BDF]] for i=BDF:ntime]
    𝐌 = vcat(Mₘₛ¹, Mₘₛ²)

    println("Done computing the multiscale space")

    # Time marching
    let 
      U₀ = setup_initial_condition(u₀, (Λₜ, Λ[1]), fine_scale_space);
      global U = zero(U₀)
      t = 0.0
      for i=1:BDF-1
        dlcache = get_dl_cache(i)            
        # Execute the BDF step
        fcache = fine_scale_space, Λₜ
        cache = dlcache, fcache
        U₁ = BDFk!(cache, t, U₀, Δt, 𝐊[i], 𝐌[i], fₙ!, i) 
        U₀ = hcat(U₁,U₀)
        t += Δt    
      end
      dlcache = get_dl_cache(BDF) 
      for i=BDF:ntime
        # Execute the BDF step
        fcache = fine_scale_space, Λₜ
        cache = dlcache, fcache
        U₁ = BDFk!(cache, t, U₀, Δt, 𝐊[i], 𝐌[i], fₙ!, BDF) 
        # Update the time step
        U₀[:, 2:BDF] = U₀[:, 1:BDF-1]
        U₀[:,1] = U₁    
        t += Δt    
      end
      U = U₀[:,1]

      U_fine_scale = Λ[ntime+1]*U
      
      # Compute the errors
      dΩ = Measure(get_triangulation(Uₕ), qorder)
      uₘₛ = FEFunction(Uₕ, U_fine_scale)    
      e = uₕ - uₘₛ
      L²Error[itr] = sqrt(sum(∫(e*e)dΩ));
      H¹Error[itr] = sqrt(sum(∫(∇(e)⋅∇(e))dΩ));      
    end

    println("Done nc = "*string(nc))
  end
  println("Done l = "*string(l))
  Plots.plot!(plt, 1 ./N, L²Error, label="(p="*string(p)*"), L \$^2\$ (l="*string(l)*")", lw=1, ls=:solid)
  Plots.plot!(plt1, 1 ./N, H¹Error, label="(p="*string(p)*"), Energy (l="*string(l)*")", lw=1, ls=:solid)
  Plots.scatter!(plt, 1 ./N, L²Error, label="", markersize=2)
  Plots.scatter!(plt1, 1 ./N, H¹Error, label="", markersize=2, legend=:best)
end

Plots.plot!(plt1, 1 ./N, (1 ./N).^(p+2), label="Order "*string(p+2), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);
Plots.plot!(plt, 1 ./N, (1 ./N).^(p+3), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);