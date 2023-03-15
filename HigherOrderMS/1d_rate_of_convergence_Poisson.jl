include("HigherOrderMS.jl");

# Create empty plots
plt = plot()
plt1 = plot()

#=
Problem data
=#
domain = (0.0,1.0)
D₁(x) = 1.0 # Smooth Diffusion coefficient
# D₁(x) = (2 + cos(2π*x/(2^-6)))^-1 # Oscillatory Diffusion coefficient
f(x) = (π)^2*sin(π*x[1])
bvals = [0.0,0.0];
u(x) = sin(π*x[1])
∇u(x) = π*cos(π*x[1])

p = 1
q = 1
N = [1,2,4,8,16,32,64]
nf = 2^16
qorder = 2

L²Error = zeros(Float64,size(N));
H¹Error = zeros(Float64,size(N));

#=
Solve the full problem once
=#
# Use Gridap to construct the space
model = CartesianDiscreteModel(domain, (nf,));
U0 = TestFESpace(Triangulation(model), ReferenceFE(lagrangian, Float64, q), conformity=:H1, dirichlet_tags="boundary");
U = TrialFESpace(bvals, U0);

stima = assemble_stiffness_matrix(domain, D, q, nf, qorder);
loadvec = assemble_load_vector(domain, f, q, nf, qorder);
fullnodes = 1:q*nf+1;
bnodes = [1, q*nf+1];
freenodes = setdiff(fullnodes, bnodes);
sol_ϵ = (stima[freenodes,freenodes])\(loadvec[freenodes]-stima[freenodes,bnodes]*bvals);


for l=[7,8]
  fill!(L²Error, 0.0)
  fill!(H¹Error, 0.0)
  for (nc,itr) in zip(N, 1:lastindex(N))
    let
      patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));

      # Compute MS bases
      basis_vec_ms = compute_ms_basis(domain, D, f, (q,p), (nf,nc), l, patch_indices_to_global_indices, qorder, [1,q*nf+1], [0.0,0.0]);

      # Solve the problem
      stima = assemble_stiffness_matrix(domain, D, q, nf, qorder);
      Kₘₛ = basis_vec_ms'*stima*basis_vec_ms;
      loadvec = assemble_load_vector(domain, f, q, nf, qorder);
      Fₘₛ = basis_vec_ms'*loadvec;
      sol = Kₘₛ\Fₘₛ

      # Obtain the solution in the fine scale for plotting
      sol_fine_scale = get_solution(sol, basis_vec_ms, nc, p);

      # Compute the errors
      dΩ = Measure(get_triangulation(U), qorder)
      uₕ = FEFunction(U, sol_ϵ)
      uₘₛ = FEFunction(U, sol_fine_scale[freenodes])    
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