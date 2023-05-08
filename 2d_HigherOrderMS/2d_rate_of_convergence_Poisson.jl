include("2d_HigherOrderMS.jl")

# Problem description
domain = (0.0, 1.0, 0.0, 1.0)
A(x) = 1.0
f(x) = 2π^2*sin(π*x[1])*sin(π*x[2]);

# Empty plots for convergence rates
plt = plot()
plt1 = plot()

function get_reference_solution(domain::Tuple, nf::Int64, q::Int64, A, f)
  model = simplexify(CartesianDiscreteModel(domain, (nf,nf)))
  reffe = ReferenceFE(lagrangian,Float64,q)
  V0 = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
  U = TrialFESpace(V0,0)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,4)
  a(u,v) = ∫( A*∇(v)⊙∇(u) )dΩ
  b(v) = ∫( v*f )dΩ
  op = AffineFEOperator(a,b,U,V0)
  uϵ = solve(op)
  uh = zeros(Float64, num_free_dofs(U)+num_dirichlet_dofs(U))
  freedofs = get_local_indices(model, "interior")
  uh[freedofs] = get_free_dof_values(uϵ)
  uh
end

nf = 2^9
q = 1
uh = get_reference_solution(domain, nf, q, A, f);

p = 3
N = [2^0, 2^1, 2^2, 2^3, 2^4]
L²Error = zeros(Float64,size(N));
H¹Error = zeros(Float64,size(N));

for l = [5,6]
  for (nc,itr) in zip(N, 1:lastindex(N))
    let 
      Ωms = MultiScaleTriangulation(domain, nf, nc, l);
      println("Built the MultiScaleTriangulation...")

      D = CellField(A, Ωms.Ωf)
      Ums = MultiScaleFESpace(Ωms, q, p, D, 4)
      println("Built the MultiScaleFESpace...")

      Fϵ = assemble_loadvec(Ums.Uh, f, 4);
      Kϵ = assemble_stima(Ums.Uh, D, 4);

      # Use the new bases to transform the matrix and vector to the multiscale space.
      basis_vec_ms = Ums.basis_vec_ms
      Kₘₛ = basis_vec_ms'*Kϵ*basis_vec_ms;
      Fₘₛ = basis_vec_ms'*Fϵ;
      sol = Kₘₛ\Fₘₛ
      ums = basis_vec_ms*sol;
      println("Computed the multiscale solution...")      

      # Get the Gridap version for visualization
      uH = FEFunction(Ums.Uh, ums);
      uϵ = FEFunction(Ums.Uh, uh)
      e = uH - uϵ
      dΩ = Measure(Ωms.Ωf, 4)
      L²Error[itr] = sqrt(sum(∫(e*e)dΩ));
      H¹Error[itr] = sqrt(sum(∫(D*∇(e)⋅∇(e))dΩ));
      println("Computed the error...")

      println("Done nc = "*string(nc))
    end
  end

  println("Done l = "*string(l)*"\n")
  plot!(plt, 1 ./N, L²Error, label="(p="*string(p)*"), L² (l="*string(l)*")", lw=2)
  plot!(plt1, 1 ./N, H¹Error, label="(p="*string(p)*"), Energy (l="*string(l)*")", lw=2)
  scatter!(plt, 1 ./N, L²Error, label="", markersize=2)
  scatter!(plt1, 1 ./N, H¹Error, label="", markersize=2, legend=:best)
end

plot!(plt1, 1 ./N, (1 ./N).^(p+2), label="Order "*string(p+2), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)
  plot!(plt, 1 ./N, (1 ./N).^(p+3), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)