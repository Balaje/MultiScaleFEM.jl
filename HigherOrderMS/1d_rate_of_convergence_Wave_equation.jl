include("HigherOrderMS.jl");
include("corrected_basis.jl");

#=
Problem data
=#
domain = (0.0,1.0)
## Random wave speed
Neps = 2^7
nds_micro = LinRange(domain[1], domain[2], Neps+1)
wave_speed_micro = 0.5 .+ 4.5*rand(Neps+1)
function _D(x::Float64, nds_micro::AbstractVector{Float64}, diffusion_micro::Vector{Float64})
  n = size(nds_micro, 1)
  for i=1:n
    if(nds_micro[i] ≤ x ≤ nds_micro[i+1])      
      return diffusion_micro[i+1]
    else
      continue
    end 
  end
end
c²(x; nds_micro = nds_micro, diffusion_micro = wave_speed_micro) = _D(x[1], nds_micro, diffusion_micro)
## Oscillatory wave speed
# c²(x) = (0.25 + 0.125*cos(2π*x[1]/2^-5))^-1
# c²(x) = (0.25 + 0.125*cos(2π*x[1]/2e-2))^-1
## Constant wave speed
# c²(x) = 1.0
u₀(x) = 0.0
# uₜ₀(x) = π*sin(π*x[1])
uₜ₀(x) = 0.0

# f(x,t) = 0.0
f(x,t) = sin(π*x[1])*sin(t)

# Problem parameters - fine scale
nf = 2^15
q = 1
qorder = 4
nds_fine = LinRange(domain[1], domain[2], q*nf+1)
# Temporal parameters
Δt = 10^-3
tf = 1.5
ntime = ceil(Int, tf/Δt)

# Solve the fine scale problem for exact solution
fine_scale_space = FineScaleSpace(domain, q, qorder, nf)
stima = assemble_stiffness_matrix(fine_scale_space, c²)
massma = assemble_mass_matrix(fine_scale_space, x->1.0)
fullnodes = 1:q*nf+1;
bnodes = [1, q*nf+1];
freenodes = setdiff(fullnodes, bnodes);
function fₙϵ!(cache, tₙ::Float64)
  fspace, freenodes = cache
  F = assemble_load_vector(fspace, y->f(y,tₙ))
  F[freenodes]
  #zeros(Float64, length(freenodes))
end
# Time marching
let 
  U₀ = u₀.(nds_fine[freenodes])
  V₀ = uₜ₀.(nds_fine[freenodes])
  global U = zero(U₀)
  cache = fine_scale_space, freenodes
  #= Crank Nicolson
  t = 0.0
  for i=1:ntime
    U₁, V₁ = CN!(cache, t, U₀, V₀, Δt, stima[freenodes,freenodes], massma[freenodes,freenodes], fₙϵ!)
    U₀, V₀ = U₁, V₁
    (i%100 == 0) && print("Done t = "*string(t)*"\n")
    t += Δt
  end
  U = U₀ # Final time solution   
  =#
  # Use the Newmark-beta scheme and choose β=0.25, γ=0.5. This will choose the implicit Crank Nicolson Method
  t = 0.0
  U₁ = NM1!(cache, U₀, V₀, Δt, stima[freenodes,freenodes], massma[freenodes,freenodes], fₙϵ!, 0.25, 0.5)
  t += Δt
  for i=2:ntime
    U = NM!(cache, t, U₁, U₀, Δt, stima[freenodes,freenodes], massma[freenodes,freenodes], fₙϵ!, 0.25, 0.5)
    U₀ = U₁
    U₁ = U
    t += Δt
  end          
end
Uₕ = TrialFESpace(fine_scale_space.U, 0.0)
uₕ = FEFunction(Uₕ, vcat(0.0,U,0.0))

##### Now begin solving using the multiscale method #####
# Create empty plots
N = [1,2,4,8,16,32,64]
# plt = plot();
# plt1 = plot();
p = 3;
L²Error = zeros(Float64,size(N));
H¹Error = zeros(Float64,size(N));
# Define the projection of the load vector onto the multiscale space
function fₙ!(cache, tₙ::Float64)
  fspace, basis_vec_ms = cache
  loadvec = assemble_load_vector(fspace, y->f(y,tₙ))
  basis_vec_ms'*loadvec
  #zeros(Float64, size(basis_vec_ms, 2))
end   

for l=[8]
# for l=[8]
  fill!(L²Error, 0.0)
  fill!(H¹Error, 0.0)
  for (nc,itr) in zip(N, 1:lastindex(N))
    let
      # Obtain the map between the coarse and fine scale
      patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
      # Compute the multiscale basis
      basis_vec_ms = compute_ms_basis(fine_scale_space, c², p, nc, l, patch_indices_to_global_indices);
      # Assemble the stiffness, mass matrices
      Kₘₛ = basis_vec_ms'*stima*basis_vec_ms
      Mₘₛ = basis_vec_ms'*massma*basis_vec_ms   
      # Add the corrected version of the basis
      KLΛ = stima, massma*basis_vec_ms, Mₘₛ
      l′ = l+5;
      # basis_vec_ms′ = compute_corrected_basis_function(fine_scale_space, KLΛ, p, nc, l, l′)      
      basis_vec_ms′ = basis_vec_ms
      Kₘₛ′ = basis_vec_ms′'*stima*basis_vec_ms′
      Mₘₛ′ = basis_vec_ms′'*massma*basis_vec_ms′
      # basis_vec_ms = basis_vec_ms′
      # Kₘₛ = Kₘₛ′
      # Mₘₛ = Mₘₛ′
      # Time marching
      let 
        U₀ = setup_initial_condition(u₀, basis_vec_ms′, fine_scale_space)
        V₀ = setup_initial_condition(uₜ₀, basis_vec_ms′, fine_scale_space)
        global U = zero(U₀)
        cache = fine_scale_space, basis_vec_ms
        #= # Crank Nicolson Method
        t = 0.0
        for i=1:ntime
          U₁, V₁ = CN!(cache, t, U₀, V₀, Δt, Kₘₛ, Mₘₛ, fₙ!)
          U₀, V₀ = U₁, V₁
          t += Δt
        end
        U = U₀ # Final time solution   
        =#
        # Leap-frog method from the Newmark scheme with β=0, γ=0.25
        # Crank Nicolson Method with β = 1/4, γ = 1/2
        t = 0.0
        U₁ = NM1!(cache, U₀, V₀, Δt, Kₘₛ′, Mₘₛ′, fₙ!, 0.25, 0.5)
        t += Δt
        for i=2:ntime
          U = NM!(cache, t, U₁, U₀, Δt, Kₘₛ′, Mₘₛ′, fₙ!, 0.25, 0.5)
          U₀ = U₁
          U₁ = U
          t += Δt
        end                
      end
      U_fine_scale = basis_vec_ms′*U
      
      # Compute the errors
      dΩ = Measure(get_triangulation(Uₕ), qorder)
      uₘₛ = FEFunction(Uₕ, U_fine_scale)    
      e = uₕ - uₘₛ
      L²Error[itr] = sqrt(sum(∫(e*e)dΩ));
      H¹Error[itr] = sqrt(sum(∫(∇(e)⋅∇(e))dΩ));
      
      println("Done nc = "*string(nc))
    end
  end
  println("Done l = "*string(l))
  plot!(plt, 1 ./N, L²Error, label="(p="*string(p)*"), L² (l="*string(l)*")", lw=2)
  plot!(plt1, 1 ./N, H¹Error, label="(p="*string(p)*"), Energy (l="*string(l)*")", lw=2)
  scatter!(plt, 1 ./N, L²Error, label="", markersize=2)
  scatter!(plt1, 1 ./N, H¹Error, label="", markersize=2, legend=:best)
end 

plot!(plt1, 1 ./N, (1 ./N).^(p+2), label="Order "*string(p+2), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);
plot!(plt, 1 ./N, (1 ./N).^(p+3), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);

# Also plot the reference lines with slope 2,3 to check the reduced rate
# plot!(plt, 1 ./N, (1 ./N).^2, label="Order 2", ls=:dash, lc=:black, xaxis=:log10, yaxis=:log10);
# plot!(plt, 1 ./N, (1 ./N).^3, label="Order 3", ls=:dash, lc=:black, xaxis=:log10, yaxis=:log10);
# plot!(plt1, 1 ./N, (1 ./N).^1, label="Order 1", ls=:dash, lc=:black, xaxis=:log10, yaxis=:log10);
# plot!(plt1, 1 ./N, (1 ./N).^2, label="Order 2", ls=:dash, lc=:black, xaxis=:log10, yaxis=:log10);