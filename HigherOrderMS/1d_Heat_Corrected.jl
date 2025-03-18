include("./src/HigherOrderMS.jl");

#=
Problem data
=#

# using Quadmath
# T₁ = Float128

using DoubleFloats
T₁ = Double64
domain = T₁.((0.0,1.0))
# Random diffusion coefficient
Neps = 2^7
nds_micro = LinRange(domain[1], domain[2], Neps+1)
diffusion_micro = 0.5 .+ (1-0.5)*rand(T₁,Neps+1)
function _D(x::T, nds_micro::AbstractVector{T}, diffusion_micro::Vector{T1}) where {T<:Number, T1<:Number}
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
f(x,t) = T₁(10*sin(π*x[1])*(sin(t))^4)
u₀(x) = T₁(0.0)

# Spatial discretization parameters
(length(ARGS)==5) && begin (nf, nc, p, l, ntimes) = parse.(Int64, ARGS) end
if(length(ARGS)==0)
  nf = 2^9;
  p = 1;
  nc = 2^3;  
  l = 5; 
  ntimes = 1;
end

# Temporal discretization parameters
Δt = 2^-7
tf = 1.0
ntime = ceil(Int, tf/Δt)
BDF = 4

println(" ")
println("Computing reference solution...")
println(" ")

# Solve the fine scale problem onfce for exact solution
fine_scale_space = FineScaleSpace(domain, 1, 6, nf; T=T₁)
nds_fine = LinRange(domain[1], domain[2], nf+1)
stima = assemble_stiffness_matrix(fine_scale_space, A)
massma = assemble_mass_matrix(fine_scale_space, x->1.0)
fullnodes = 1:nf+1;
bnodes = [1, nf+1];
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
    println("Done t = "*string(t))
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
    (i%(ntime/2^4) == 0) && println("Done t = "*string(t))
  end
  Uex = U₀[:,1] # Final time solution
end
Uₕ = TrialFESpace(fine_scale_space.U, 0.0)
uₕ = FEFunction(Uₕ, vcat(0.0,Uex,0.0))

println(" ")
println("Solving MS problem...")

###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
# Begin solving using the new multiscale method and compare the convergence rates #
###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
# Define the projection of the load vector onto the multiscale space
function fₙ!(cache, tₙ::Float64)
  # "A Computationally Efficient Method"
  fspace, basis_vec_ms, basis_vec_ms₂ = cache
  loadvec = assemble_load_vector(fspace, y->f(y,tₙ))
  [basis_vec_ms₂'*loadvec; basis_vec_ms'*loadvec]
end   

# Obtain the map between the coarse and fine scale
patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (1,p));

# Obtain the basis functions
basis_vec_ms₁ = compute_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices; T=T₁);

# Compute the stabilized basis functions
isStab = true
(isStab) && begin println("Stabilization on ..."); println(""); end
if(nc > 1 && isStab)
  γ = Cˡιₖ(fine_scale_space, A, p, nc, l; T=T₁);
  basis_vec_ms₁[:, 1:(p+1):(p+1)*nc] = γ;    
end      

# Compute the additional correction basis
p′ = p
basis_vec_ms₂ = compute_correction_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices, p′; T=T₁, ntimes=ntimes, isStab=isStab);      

# Assemble the stiffness, mass matrices
Kₘₛ = basis_vec_ms₁'*stima*basis_vec_ms₁; 
Mₘₛ = basis_vec_ms₁'*massma*basis_vec_ms₁; 
Kₘₛ′ = basis_vec_ms₂'*stima*basis_vec_ms₂; 
Mₘₛ′ = basis_vec_ms₂'*massma*basis_vec_ms₂; 
Lₘₛ = basis_vec_ms₂'*massma*basis_vec_ms₁
Pₘₛ = basis_vec_ms₂'*stima*basis_vec_ms₁

𝐌 = [Mₘₛ′ Lₘₛ; Lₘₛ'  Mₘₛ]
𝐊 = [Kₘₛ′ Pₘₛ; Pₘₛ' Kₘₛ]

# Time marching
let         
  # Project the initial condition onto the multiscale space
  U₀ = [zeros(T₁, ntimes*(p′+1)*nc); setup_initial_condition(u₀, basis_vec_ms₁, fine_scale_space)]
  fcache = fine_scale_space, basis_vec_ms₁, basis_vec_ms₂
  global U = zero(U₀)  
  t = 0.0
  # Starting BDF steps (1...k-1) 
  for i=1:BDF-1
    dlcache = get_dl_cache(i)
    cache = dlcache, fcache
    U₁ = BDFk!(cache, t, U₀, Δt, 𝐊, 𝐌, fₙ!, i)
    U₀ = hcat(U₁, U₀)
    t += Δt   
    println("Done t = "*string(t))       
  end
  # Remaining BDF steps
  dlcache = get_dl_cache(BDF)
  cache = dlcache, fcache
  for i=BDF:ntime
    U₁ = BDFk!(cache, t+Δt, U₀, Δt, 𝐊, 𝐌, fₙ!, BDF)
    U₀[:,2:BDF] = U₀[:,1:BDF-1]
    U₀[:,1] = U₁
    t += Δt  
    (i%(ntime/2^4) == 0) && println("Done t = "*string(t))        
  end
  U = U₀[:,1] # Final time solution
end      

# Construct the corrected solution
U₁ = U[ntimes*(p′+1)*nc+1:end] 
dU₁ = U[1:ntimes*(p′+1)*nc]
U_fine_scale = basis_vec_ms₁*U₁+ basis_vec_ms₂*dU₁

# Compute the errors
dΩ = Measure(get_triangulation(Uₕ), 6)
uₘₛ = FEFunction(Uₕ, U_fine_scale)    
e = uₕ - uₘₛ
L²Error = sqrt(sum(∫(e*e)dΩ));
H¹Error = sqrt(sum(∫(∇(e)⋅∇(e))dΩ));

println(" ")
println("$p \t $nc \t $l \t $L²Error \t $H¹Error")
