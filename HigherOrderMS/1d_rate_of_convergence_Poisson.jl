include("HigherOrderMS.jl");

# Create empty plots
plt = plot()
plt1 = plot()

#=
Problem data
=#
D₁(x) = 1.0 # Smooth Diffusion coefficient
# D₁(x) = (2 + cos(2π*x/(2^-6)))^-1 # Oscillatory Diffusion coefficient
f(x) = (π)^2*sin(π*x[1])
u(x) = sin(π*x[1])
∇u(x) = π*cos(π*x[1])
domain = (0.0,1.0)

p = 1
q = 1
N = [1,2,4,8,16,32,64]
nf = 2^16
qorder = 2

L²Error = zeros(Float64,size(N))
H¹Error = zeros(Float64,size(N))

#=
Solve the full problem once
=#
KMf, _, _, U = get_saddle_point_problem(domain, D₁, f, (q,p), (nf,nc), qorder)
stima, _, loadvec = KMf
fn = 2:q*nf
sol_ϵ = vcat(0.0,stima[fn,fn]\loadvec[fn])


for l=[7,8]
  fill!(L²Error, 0.0)
  fill!(H¹Error, 0.0)
  for (nc,itr) in zip(N, 1:lastindex(N))
    let
      patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
      
      Kf, basis_vec_ms, U = compute_ms_basis((0.0,1.0), D₁, f, (q,p), (nf,nc), l, patch_indices_to_global_indices, qorder);
      stima, massma, loadvec = Kf
      
      # Solve the problem
      basis_elem_ms = BroadcastVector(getindex, Fill(basis_vec_ms,nc), coarse_indices_to_fine_indices, ms_elem);
      basis_elem_ms_t = BroadcastVector(transpose, basis_elem_ms);
      # (-) Get the multiscale stiffness matrix
      mat_el = BroadcastVector(mat_contribs, Fill(stima, nc), coarse_indices_to_fine_indices, 1:nc, nc)
      ms_elem_mats = BroadcastVector(*, basis_elem_ms_t, mat_el, basis_elem_ms);
      stima_ms = assemble_ms_matrix(ms_elem_mats, ms_elem, nc, p);
      # (-) Get the multiscale load vector  
      vec_el = BroadcastVector(vec_contribs, Fill(loadvec, nc), coarse_indices_to_fine_indices, 1:nc, nc)
      ms_elem_vecs = BroadcastVector(*, basis_elem_ms_t, vec_el);
      loadvec_ms = assemble_ms_vector(ms_elem_vecs, ms_elem, nc, p);
      # (-) Solve the problem
      sol = materialize(stima_ms)\materialize(loadvec_ms)
      sol_fine_scale = get_solution(sol, basis_vec_ms);

      dΩ = Measure(get_triangulation(U), qorder)
      uₕ = interpolate_everywhere(sol_ϵ, U)
      uₘₛ = interpolate_everywhere(sol_fine_scale[1:end-1], U)    
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