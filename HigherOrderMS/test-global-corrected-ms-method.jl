include("HigherOrderMS.jl");
include("corrected_basis.jl");

#=
Problem data
=#
domain = (0.0,1.0)
# Random diffusion coefficient
Neps = 2^8
nds_micro = LinRange(domain[1], domain[2], Neps+1)
diffusion_micro = 0.1 .+ 0.1*rand(Neps+1)
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
# A(x) = (2 + cos(2π*x[1]/2^-6))^-1 # Oscillatory diffusion coefficient
# A(x) = (2 + cos(2π*x[1]/2^0))^-1 # Smooth Diffusion coefficient
# A(x) = 1.0 # Constant diffusion coefficient
f(x,t) = sin(π*x[1])*sin(π*t)
u₀(x) = 0.0
# f(x,t) = 0.0
# u₀(x) = sin(π*x[1])

# Problem parameters
nf = 2^15
q = 1
qorder = 6
# Temporal parameters
Δt = 1e-3
tf = 1e-2
ntime = ceil(Int, tf/Δt)
BDF = 4

# Solve the fine scale problem onfce for exact solution
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
    (i%(ntime/10) ≈ 0.0) && println("Done t = "*string(t))
  end
  # Remaining BDF steps
  dlcache = get_dl_cache(BDF)
  cache = dlcache, fcache
  for i=BDF:ntime
    U₁ = BDFk!(cache, t+Δt, U₀, Δt, stima[freenodes,freenodes], massma[freenodes,freenodes], fₙϵ!, BDF)
    U₀[:,2:BDF] = U₀[:,1:BDF-1]
    U₀[:,1] = U₁
    t += Δt
    (i%(ntime/10) ≈ 0.0) && println("Done t = "*string(t))
  end
  Uex = U₀[:,1] # Final time solution
end
Uₕ = TrialFESpace(fine_scale_space.U, 0.0)
uₕ = FEFunction(Uₕ, vcat(0.0,Uex,0.0))

println(" ")
println("Solving MS problem...")
println(" ")

##### Now begin solving using the multiscale method #####
N = [1,2,4,8]
# Create empty plots
plt = Plots.plot();
plt1 = Plots.plot();
plt7_1 = Plots.plot();
plt7_2 = Plots.plot();
plt7 = Plots.plot()
p = 3;
L²Error = zeros(Float64,size(N));
H¹Error = zeros(Float64,size(N));
# Define the projection of the load vector onto the multiscale space
function fₙ!(cache, tₙ::Float64)
  fspace, basis_vec_ms, z1, z2 = cache
  loadvec = assemble_load_vector(fspace, y->f(y,tₙ))
  [z1; z2; basis_vec_ms'*loadvec]

  # "A Computationally Efficient Method"
  # fspace, basis_vec_ms, z1 = cache
  # loadvec = assemble_load_vector(fspace, y->f(y,tₙ))
  # [z1; basis_vec_ms'*loadvec]
end   

maxvals_correction = zeros(Float64, ntime)

for l=[8]
  fill!(L²Error, 0.0)
  fill!(H¹Error, 0.0)
  for (nc,itr) in zip(N, 1:lastindex(N))
    let      
      # Obtain the map between the coarse and fine scale
      patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
      # Compute the multiscale basis
      global basis_vec_ms₁ = compute_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices);
      # Assemble the stiffness, mass matrices
      Kₘₛ = basis_vec_ms₁'*stima*basis_vec_ms₁
      Mₘₛ = basis_vec_ms₁'*massma*basis_vec_ms₁       
      L₁ = massma*basis_vec_ms₁;
      L₂ = basis_vec_ms₁'*massma;
      P = get_saddle_point_problem(fine_scale_space, A, p, nc)[2]
            
      global 𝐌 = [massma[freenodes, freenodes] zero(P[freenodes,:])       -L₁[freenodes,:]; 
                     zero(P[freenodes,:])'      zeros(nc*(p+1), nc*(p+1))  zeros(nc*(p+1), nc*(p+1));
                     -L₂[:,freenodes]           zeros(nc*(p+1), nc*(p+1))     Mₘₛ];
      
      global 𝐊 = [stima[freenodes, freenodes]     P[freenodes,:]            zero(L₁[freenodes,:]);
                      P[freenodes,:]'         zeros(nc*(p+1), nc*(p+1))     zeros(nc*(p+1), nc*(p+1));
                   zero(L₂[:,freenodes])      zeros(nc*(p+1), nc*(p+1))         Kₘₛ];

      # "A Computationally Efficient Method"
      # global 𝐌 = [Mₘₛ -Mₘₛ;
      #             -Mₘₛ  Mₘₛ]
      # global 𝐊 = blockdiag(Kₘₛ, Kₘₛ);
      
          
      # Time marching
      let 
        # Project initial condition onto the multiscale space
        U₀ = [zeros(length(freenodes)); zeros(nc*(p+1)); setup_initial_condition(u₀, basis_vec_ms₁, fine_scale_space)]

        # "A Computationally Efficient Method"
        # U₀ = [zeros(nc*(p+1)); setup_initial_condition(u₀, basis_vec_ms₁, fine_scale_space)]
        # fcache = fine_scale_space, basis_vec_ms₁, zeros(nc*(p+1))        
        global U = zero(U₀)  
        t = 0.0
        # Starting BDF steps (1...k-1) 
        fcache = fine_scale_space, basis_vec_ms₁, zeros(length(freenodes)), zeros(nc*(p+1))        
        for i=1:BDF-1
          dlcache = get_dl_cache(i)
          cache = dlcache, fcache
          U₁ = BDFk!(cache, t, U₀, Δt, 𝐊, 𝐌, fₙ!, i)
          U₀ = hcat(U₁, U₀)
          t += Δt
          (i%(ntime/20) ≈ 0.0) && println("Done t = "*string(t))          
          if(nc==1) 
            Plots.plot!(plt7_1, nds_fine, basis_vec_ms₁*U₀[length(freenodes)+nc*(p+1)+1:end, 1], label="")
            Plots.plot!(plt7_2, nds_fine, [0.0; U₀[1:length(freenodes), 1]; 0.0], label="")           
          end
        end
        # Remaining BDF steps
        dlcache = get_dl_cache(BDF)
        cache = dlcache, fcache
        for i=BDF:ntime
          U₁ = BDFk!(cache, t+Δt, U₀, Δt, 𝐊, 𝐌, fₙ!, BDF)
          U₀[:,2:BDF] = U₀[:,1:BDF-1]
          U₀[:,1] = U₁
          t += Δt
          (i%(ntime/20) ≈ 0.0) && println("Done t = "*string(t))          
          if(nc==1) 
            Plots.plot!(plt7_1, nds_fine, basis_vec_ms₁*U₀[length(freenodes)+nc*(p+1)+1:end, 1], label="")
            Plots.plot!(plt7_2, nds_fine, [0.0; U₀[1:length(freenodes), 1]; 0.0], label="")
          end
        end
        U = U₀[:,1] # Final time solution
      end
      U_fine_scale = (basis_vec_ms₁*U[length(freenodes)+nc*(p+1)+1:end]) - [0; U[1:length(freenodes)]; 0]

      # "A Computationally Efficient Method"
      # U_sol = reshape(U, nc*(p+1), 2)
      # U_fine_scale = basis_vec_ms₁*(U_sol[:,2] - U_sol[:,1])
      
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
  Plots.plot!(plt, 1 ./N, L²Error, label="(p="*string(p)*"), L² (l="*string(l)*")", lw=2)
  Plots.plot!(plt1, 1 ./N, H¹Error, label="(p="*string(p)*"), Energy (l="*string(l)*")", lw=2)
  Plots.scatter!(plt, 1 ./N, L²Error, label="", markersize=2)
  Plots.scatter!(plt1, 1 ./N, H¹Error, label="", markersize=2, legend=:best)
end 

Plots.plot!(plt1, 1 ./N, (1 ./N).^(p+2), label="Order "*string(p+2), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);
Plots.plot!(plt, 1 ./N, (1 ./N).^(p+3), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);

# Plot the rates along with the diffusion coefficient
plt2 = Plots.plot(plt, plt1, layout=(1,2))
plt3 = Plots.plot(nds_fine, A.(nds_fine), lw=2, label="A(x)")
plt5 = Plots.plot(plt3, plt2, layout=(2,1))

# Switch variables to global and plot
plt4 = Plots.plot()
nc = N[end]
U_sol = (basis_vec_ms₁*U[length(freenodes)+nc*(p+1)+1:end]) - [0; U[1:length(freenodes)]; 0]
Plots.plot!(plt4, nds_fine, U_sol, label="Multiscale solution", lw=2)
Plots.plot!(plt4, nds_fine, vcat(0.0, Uex, 0.0), label="Reference Solution", lw=1, ls=:dash, lc=:black)

plt6 = Plots.plot(plt, plt1, plt3, plt4, layout=(2,2))