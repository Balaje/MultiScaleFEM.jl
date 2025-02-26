include("HigherOrderMS.jl");
include("corrected_basis.jl");

plt = Plots.plot();
plt1 = Plots.plot();

#=
Problem data
=#
T₁ = Double64
domain = T₁.((0.0,1.0))
# Random diffusion coefficient
Neps = 2^7
nds_micro = LinRange(domain[1], domain[2], Neps+1)
diffusion_micro = 0.2 .+ (1-0.2)*rand(T₁,Neps+1)
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
# A(x) = (2 + cos(2π*x[1]/2^1))^-1 # Oscillatory diffusion coefficient
# A(x) = (2 + cos(2π*x[1]/2^0))^-1 # Smooth Diffusion coefficient
# A(x) = 0.5 # Constant diffusion coefficient
f(x,t) = T₁(10*sin(π*x[1])*(sin(t))^4)
u₀(x) = T₁(0.0)
# f(x,t) = 0.0
# u₀(x) = sin(π*x[1])

# Problem parameters
nf = 2^9
q = 1
qorder = 6
# Temporal parameters
Δt = 2^-7
tf = 1.0
ntime = ceil(Int, tf/Δt)
BDF = 4

# Solve the fine scale problem onfce for exact solution
fine_scale_space = FineScaleSpace(domain, q, qorder, nf; T=T₁)
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
println("Solving new MS problem...")
println(" ")

N = 2 .^(0:5)
# Create empty plots
plt = Plots.plot();
plt1 = Plots.plot();
p = 1;

###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
# Begin solving using the new multiscale method and compare the convergence rates #
###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
L²Error = zeros(T₁,size(N));
H¹Error = zeros(T₁,size(N));
# Define the projection of the load vector onto the multiscale space
function fₙ!(cache, tₙ::Float64)
  # "A Computationally Efficient Method"
  fspace, basis_vec_ms, basis_vec_ms₂ = cache
  loadvec = assemble_load_vector(fspace, y->f(y,tₙ))
  [basis_vec_ms₂'*loadvec; basis_vec_ms'*loadvec]
end   

for ntimes = [1]
for p′ = [p]
for l = [3,5]
  fill!(L²Error, 0.0)
  fill!(H¹Error, 0.0)
  for (nc,itr) in zip(N, 1:lastindex(N))
    global lw = 1
    global ls = :dash
    global isStab = false
    let            
      nc′ = nc
      # Obtain the map between the coarse and fine scale
      patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
      global basis_vec_ms₁ = compute_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices; T=T₁);
      # Compute the stabilized basis functions
      # if(nc > 1)
      #   γ = Cˡιₖ(fine_scale_space, A, p, nc, l; T=T₁);
      #   basis_vec_ms₁[:, 1:(p+1):(p+1)*nc] = γ;
      #   global lw = 2
      #   global ls = :solid
      #   global isStab = true
      # end      

      # Compute the multiscale basis
      patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc′, nf, l, (q,p′));
      global basis_vec_ms₂ = compute_l2_orthogonal_basis(fine_scale_space, A, p, nc′, l, patch_indices_to_global_indices, p′; T=T₁, ntimes=ntimes, isStab=isStab);      

      # Assemble the stiffness, mass matrices
      Kₘₛ = basis_vec_ms₁'*stima*basis_vec_ms₁; Mₘₛ = basis_vec_ms₁'*massma*basis_vec_ms₁; 
      Kₘₛ′ = basis_vec_ms₂'*stima*basis_vec_ms₂; Mₘₛ′ = basis_vec_ms₂'*massma*basis_vec_ms₂; 
      Lₘₛ = basis_vec_ms₂'*massma*basis_vec_ms₁
      Pₘₛ = basis_vec_ms₂'*stima*basis_vec_ms₁
            
      global 𝐌 = [Mₘₛ′ Lₘₛ; 
                  Lₘₛ'  Mₘₛ];
      global 𝐊 = [Kₘₛ′ Pₘₛ; 
                  Pₘₛ' Kₘₛ]
      # global 𝐌 = Mₘₛ′
      # global 𝐊 = Kₘₛ′

      # sM = SchurComplementMatrix(𝐌, (nc*(p′+1), nc*(p+1)))
      # sK = SchurComplementMatrix(𝐊, (nc*(p′+1), nc*(p+1)))
      global sM = collect(𝐌);
      global sK = collect(𝐊);
                
      # Time marching
      let 
        # Project initial condition onto the multiscale space
        
        # "A Computationally Efficient Method"
        U₀ = [zeros(T₁, ntimes*(p′+1)*nc′); setup_initial_condition(u₀, basis_vec_ms₁, fine_scale_space)]
        fcache = fine_scale_space, basis_vec_ms₁, basis_vec_ms₂
        global U = zero(U₀)  
        t = 0.0
        # Starting BDF steps (1...k-1) 
        for i=1:BDF-1
          dlcache = get_dl_cache(i)
          cache = dlcache, fcache
          U₁ = BDFk!(cache, t, U₀, Δt, sK, sM, fₙ!, i)
          U₀ = hcat(U₁, U₀)
          t += Δt   
          # println("Done t = "*string(t))       
        end
        # Remaining BDF steps
        dlcache = get_dl_cache(BDF)
        cache = dlcache, fcache
        for i=BDF:ntime
          U₁ = BDFk!(cache, t+Δt, U₀, Δt, sK, sM, fₙ!, BDF)
          U₀[:,2:BDF] = U₀[:,1:BDF-1]
          U₀[:,1] = U₁
          t += Δt  
          # println("Done t = "*string(t))        
        end
        U = U₀[:,1] # Final time solution
      end      

      # "A Computationally Efficient Method"            
      U_fine_scale = basis_vec_ms₁*U[ntimes*(p′+1)*nc′+1:end] + basis_vec_ms₂*U[1:ntimes*(p′+1)*nc′]
      
      # Compute the errors
      dΩ = Measure(get_triangulation(Uₕ), qorder)
      uₘₛ = FEFunction(Uₕ, U_fine_scale)    
      e = uₕ - uₘₛ
      L²Error[itr] = sqrt(sum(∫(e*e)dΩ));
      H¹Error[itr] = sqrt(sum(∫(∇(e)⋅∇(e))dΩ));

      # println("nc = "*string(nc)*" cond(Mₘₛ) = "*string(cond(collect(Mₘₛ)))*" cond(Mₘₛ′) = "*string(cond(collect(Mₘₛ′)))*" cond(𝐌) = "*string(cond(SchurComplementMatrix(collect(𝐌 + Δt*𝐊), (nc*(p′+1), nc*(p+1))))))      
      println("nc = $nc, norm(basis_vec_ms₁) = $(norm(basis_vec_ms₁)), norm(basis_vec_ms₂) = $(norm(basis_vec_ms₂))")
    end
  end
  println("Done l = "*string(l))
  Plots.plot!(plt, 1 ./N, L²Error, label="(p="*string(p)*", q="*string(p′)*", j=$ntimes) L\$^2\$ (l="*string(l)*")", lw=lw, ls=ls)
  Plots.plot!(plt1, 1 ./N, H¹Error, label="(p="*string(p)*", q="*string(p′)*", j=$ntimes) Energy (l="*string(l)*")", lw=lw, ls=ls)
  Plots.scatter!(plt, 1 ./N, L²Error, label="", markersize=2, xaxis=:log2, yaxis=:log10)
  Plots.scatter!(plt1, 1 ./N, H¹Error, label="", markersize=2, legend=:best, xaxis=:log2, yaxis=:log10)
  
  # Plots.plot!(plt, 1 ./N, L²Error[1]*(1 ./N).^(p+2), label="Order "*string(p+2), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);
  # Plots.plot!(plt1, 1 ./N, H¹Error[1]*(1 ./N).^(p+3), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);  
end
println("Done q = "*string(p′)) 
end
println("Done ntimes = $ntimes")
end

ev = eigvals(collect(𝐌 + Δt*𝐊));
# plt_ev = Vector{Plots.Plot}(undef, 3);
# plt_ev[1] = Plots.plot();
# Plots.scatter!(plt_ev[1], real(ev), imag(ev), label="Eigenvalues \$N_H = "*string(N[1])*", N_{\\epsilon} = "*string(Neps)*"\$ (New Method)", msw=0.0);

# # Plot the corrected solution
# plt4 = Plots.plot()
# nc = N[end]
# p′ = 2
# U_fine_scale = basis_vec_ms₁*U[(p′+1)*δ*nc+1:end] + basis_vec_ms₂*U[1:(p′+1)*δ*nc]
# plt7_1 = Plots.plot(nds_fine, basis_vec_ms₁*U[(p′+1)*(δ)*nc+1:end])
# plt7_2 = Plots.plot(nds_fine, basis_vec_ms₂*U[1:(p′+1)*(δ)*nc])
# plt7 = Plots.plot(plt7_1, plt7_2, layout=(1,2))
# Plots.plot!(plt4, nds_fine, U_fine_scale, label="New Multiscale solution", lw=1)

println(" ")
println("Solving old MS problem...")
println(" ")

#=
###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
# Begin solving using the old multiscale method and compare the convergence rates #
###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
L²Error = zeros(T₁,size(N));
H¹Error = zeros(T₁,size(N));
# Define the projection of the load vector onto the multiscale space
function fₙ!(cache, tₙ::Float64)
  # "A Computationally Efficient Method"
  fspace, basis_vec_ms = cache
  loadvec = assemble_load_vector(fspace, y->f(y,tₙ))
  basis_vec_ms'*loadvec
end   

for l=[N[end]]
  fill!(L²Error, 0.0)
  fill!(H¹Error, 0.0)
  for (nc,itr) in zip(N, 1:lastindex(N))
    let      
      # Obtain the map between the coarse and fine scale
      patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
      # Compute the multiscale basis
      global basis_vec_ms = compute_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices, T=T₁);            
      # Assemble the stiffness, mass matrices
      global Kₘₛ = basis_vec_ms'*stima*basis_vec_ms; 
      global Mₘₛ = basis_vec_ms'*massma*basis_vec_ms;                         
      # Time marching
      let 
        # Project initial condition onto the multiscale space        
        # "A Computationally Efficient Method"
        U₀ = setup_initial_condition(u₀, basis_vec_ms, fine_scale_space)
        fcache = fine_scale_space, basis_vec_ms
        global U = zero(U₀)  
        t = 0.0
        # Starting BDF steps (1...k-1) 
        for i=1:BDF-1
          dlcache = get_dl_cache(i)
          cache = dlcache, fcache
          U₁ = BDFk!(cache, t, U₀, Δt, collect(Kₘₛ), collect(Mₘₛ), fₙ!, i)
          U₀ = hcat(U₁, U₀)
          t += Δt        
        end
        # Remaining BDF steps
        dlcache = get_dl_cache(BDF)
        cache = dlcache, fcache
        for i=BDF:ntime
          U₁ = BDFk!(cache, t+Δt, U₀, Δt, collect(Kₘₛ), collect(Mₘₛ), fₙ!, BDF)
          U₀[:,2:BDF] = U₀[:,1:BDF-1]
          U₀[:,1] = U₁
          t += Δt          
        end
        U = U₀[:,1] # Final time solution
      end      

      # "A Computationally Efficient Method"      
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
  Plots.plot!(plt, 1 ./N, L²Error, label="(p="*string(p)*"), L\$^2\$ (l="*string(l)*")", lw=3, ls=:dash)
  Plots.plot!(plt1, 1 ./N, H¹Error, label="(p="*string(p)*"), Energy (l="*string(l)*")", lw=3, ls=:dash)
  Plots.scatter!(plt, 1 ./N, L²Error, label="", markersize=2)
  Plots.scatter!(plt1, 1 ./N, H¹Error, label="", markersize=2, legend=:best)
end 
=#
Plots.plot!(plt1, 1 ./N, H¹Error[1]*(1 ./N).^(p+2), label="Order "*string(p+2), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);
Plots.plot!(plt, 1 ./N, L²Error[1]*(1 ./N).^(p+3), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);
#Plots.plot!(plt1, 1 ./N, H¹Error[1]*(1 ./N).^(2.5), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);
#Plots.plot!(plt, 1 ./N, L²Error[1][1]*(1 ./N).^(2.5), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);


# Plot the rates along with the diffusion coefficient
# plt2 = Plots.plot(plt, plt1, layout=(1,2))
# plt3 = Plots.plot(nds_fine, A.(nds_fine), lw=2, label="A(x)")
# plt5 = Plots.plot(plt3, plt2, layout=(2,1))

# Switch variables to global and plot
# U_fine_scale = basis_vec_ms*U
# Plots.plot!(plt4, nds_fine, U_fine_scale, label="Old Multiscale solution", lw=1.5, ls=:dash)
# Plots.plot!(plt4, nds_fine, [0.0; Uex; 0.0], label="Reference solution", lw=2, ls=:dot)

# plt6 = Plots.plot(plt, plt1, plt3, plt4, layout=(2,2))