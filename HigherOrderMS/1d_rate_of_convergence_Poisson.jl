include("HigherOrderMS.jl");

# Create empty plots
plt = plot()
plt1 = plot()

#=
Problem data
=#
domain = (0.0,1.0)
D(x) = 1.0 # Smooth Diffusion coefficient
# D(x) = (2 + cos(2π*x/(2^-6)))^-1 # Oscillatory Diffusion coefficient
f(x) = (π)^2*cos(π*x[1])
bvals = [1.0,-1.0];

# Fine scale parameters
q = 1
nf = 2^16
qorder = 2

# Use Gridap to construct the space
fine_scale_space = FineScaleSpace(domain, q, qorder, nf)

# Solve the full problem once
stima = assemble_stiffness_matrix(fine_scale_space, D)
loadvec = assemble_load_vector(fine_scale_space, f)
fullnodes = 1:q*nf+1;
bnodes = [1, q*nf+1];
freenodes = setdiff(fullnodes, bnodes);
sol_ϵ = (stima[freenodes,freenodes])\(loadvec[freenodes]-stima[freenodes,bnodes]*bvals);
U = fine_scale_space.U
uₕ = FEFunction(U, vcat(bvals[1],sol_ϵ,bvals[2]))

# Coarse scale parameters
p = 2
N = [1,2,4,8,16,32,64]
L²Error = zeros(Float64,size(N));
H¹Error = zeros(Float64,size(N));

for l=[7,8,9]
  fill!(L²Error, 0.0)
  fill!(H¹Error, 0.0)
  for (nc,itr) in zip(N, 1:lastindex(N))
    let
      # Compute the map between coarse and fine scale
      patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
      # Compute MS bases
      basis_vec_ms = compute_ms_basis(fine_scale_space, D, p, nc, l, patch_indices_to_global_indices)
      # Compute boundary contributions
      Pₕug = compute_boundary_correction_matrix(fine_scale_space, D, p, nc, l, patch_indices_to_global_indices);
      boundary_contrib = apply_boundary_correction(Pₕug, bnodes, bvals, patch_indices_to_global_indices, p, nc, l, fine_scale_space);
      # Solve the problem
      # stima = assemble_stiffness_matrix(fine_scale_space, D)
      # loadvec = assemble_load_vector(fine_scale_space, f)
      Kₘₛ = basis_vec_ms'*stima[:,freenodes]*basis_vec_ms[freenodes,:];
      Fₘₛ = basis_vec_ms'*loadvec - basis_vec_ms'*(stima[:,bnodes]*bvals);
      sol2 = Kₘₛ\Fₘₛ;
      #Apply the boundary correction
      sol_fine_scale = basis_vec_ms*sol2 + boundary_contrib

      # Compute the errors
      dΩ = Measure(get_triangulation(U), qorder)
      uₘₛ = FEFunction(U, sol_fine_scale)    
      e = uₕ - uₘₛ
      L²Error[itr] = sqrt(sum(∫(e*e)dΩ));
      H¹Error[itr] = sqrt(sum(∫(D*∇(e)⋅∇(e))dΩ));

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