include("HigherOrderMS.jl");

# Create empty plots
plt = plot()
plt1 = plot()

#=
Problem data
=#
domain = (0.0,1.0)
A(x) = 1.0 # Smooth Diffusion coefficient
f(x,t) = 0.0
u₀(x) = sin(π*x[1])

# Problem parameters
nf = 2^16
q = 1
qorder = 4
# Temporal parameters
Δt = 10^-3
tf = 1.0
ntime = ceil(Int, tf/Δt)
BDF = 4

# Solve the fine scale problem once for exact solution
fine_scale_space = FineScaleSpace(domain, q, qorder, nf)
nds_fine = LinRange(domain[1], domain[2], q*nf+1)
stima = assemble_stiffness_matrix(fine_scale_space, A)
massma = assemble_mass_matrix(fine_scale_space, x->1.0)
fullnodes = 1:q*nf+1;
bnodes = [1, q*nf+1];
freenodes = setdiff(fullnodes, bnodes);
function fₙϵ!(cache, tₙ::Float64)
  fspace, freenodes = cache
  F = assemble_load_vector(fspace, y->f(y,tₙ))
  F[freenodes]
end
let 
  U₀ = u₀.(nds_fine[freenodes])
  global U = zero(U₀)  
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
  U = U₀[:,1] # Final time solution
end
Uₕ = TrialFESpace(fine_scale_space.U, 0.0)
uₕ = FEFunction(Uₕ, vcat(0.0,U,0.0))

##### Now begin solving using the multiscale method #####
N = [1,2,4,8,16,32,64]
p = 1
L²Error = zeros(Float64,size(N));
H¹Error = zeros(Float64,size(N));
# Define the projection of the load vector onto the multiscale space
function fₙ!(cache, tₙ::Float64)
  fspace, basis_vec_ms = cache
  loadvec = assemble_load_vector(fspace, y->f(y,tₙ))
  basis_vec_ms'*loadvec
end   

for l=[7,8]
  fill!(L²Error, 0.0)
  fill!(H¹Error, 0.0)
  for (nc,itr) in zip(N, 1:lastindex(N))
    let
      # Obtain the map between the coarse and fine scale
      patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
      # Compute the multiscale basis
      basis_vec_ms = compute_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices);
      # Assemble the stiffness, mass matrices
      Kₘₛ = basis_vec_ms'*stima*basis_vec_ms
      Mₘₛ = basis_vec_ms'*massma*basis_vec_ms   
      let 
        # Project initial condition onto the multiscale space
        U₀ = setup_initial_condition(u₀, basis_vec_ms, fine_scale_space)  
        global U = zero(U₀)  
        t = 0.0
        # Starting BDF steps (1...k-1) 
        fcache = (fine_scale_space, basis_vec_ms) 
        for i=1:BDF-1
          dlcache = get_dl_cache(i)
          cache = dlcache, fcache
          U₁ = BDFk!(cache, t, U₀, Δt, Kₘₛ, Mₘₛ, fₙ!, i)
          U₀ = hcat(U₁, U₀)
          t += Δt
        end
        # Remaining BDF steps
        dlcache = get_dl_cache(BDF)
        cache = dlcache, fcache
        for i=BDF:ntime
          U₁ = BDFk!(cache, t+Δt, U₀, Δt, Kₘₛ, Mₘₛ, fₙ!, BDF)
          U₀[:,2:BDF] = U₀[:,1:BDF-1]
          U₀[:,1] = U₁
          t += Δt
        end
        U = U₀[:,1] # Final time solution
      end
      U_fine_scale = basis_vec_ms*U
      
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

plot!(plt1, 1 ./N, (1 ./N).^(p+2), label="Order "*string(p+2), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)
plot!(plt, 1 ./N, (1 ./N).^(p+3), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)