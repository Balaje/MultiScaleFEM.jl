include("2d_HigherOrderMS.jl");

# Problem description
domain = (0.0, 1.0, 0.0, 1.0)
A(x) = 1.0
f(x) = sin(π*x[1])*sin(π*x[2])

# Construct the triangulation of the fine-scale
nf = 2^8
model = CartesianDiscreteModel(domain, (nf,nf))
Ω_fine = Triangulation(model)
reffe = ReferenceFE(lagrangian, Float64, 1)
Vh = TestFESpace(Ω_fine, reffe, conformity=:H1, dirichlet_tags="boundary");
Vh0 = TrialFESpace(Vh, 0.0);
dΩ_fine = Measure(Ω_fine, 5);
a(u,v) = ∫(A*(∇(v)⊙∇(u)))dΩ_fine;
b(v) = ∫(v*f)dΩ_fine;
op = AffineFEOperator(a,b,Vh0,Vh);
Uex = solve(op);

N = [1,2,4,8];
L²Error = zeros(Float64, length(N));
H¹Error = zeros(Float64, length(N));

plt1 = Plots.plot();
plt2 = Plots.plot();

let 
  V0 = TestFESpace(Ω_fine, reffe, conformity=:H1)
  global K = assemble_stima(V0, A, 5);
  global F = assemble_loadvec(V0, f, 5);
  for p=[1]
    for l=[3,6]
      for (nc,i) = zip(N, 1:length(N))
        # Construct the triangulation of the coarse-scale
        global model_coarse = CartesianDiscreteModel(domain, (nc,nc))
        Ω_coarse = Triangulation(model_coarse)
        # Obtain the coarse-to-fine map
        nsteps =  (Int64(log2(nf/nc)))
        coarse_to_fine_map = coarsen(model, nsteps); 
        global Ωₘₛ = MultiScaleTriangulation(domain, nf, nc, l);
        global Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V0, A);
        global basis_vec_ms = Vₘₛ.basis_vec_ms
        
        global Kₘₛ = basis_vec_ms'*K*basis_vec_ms;
        global fₘₛ = basis_vec_ms'*F;
        global solₘₛ = Kₘₛ\fₘₛ;
        
        Uₘₛ = basis_vec_ms*solₘₛ;
        Uₘₛʰ = FEFunction(Vₘₛ.Uh, Uₘₛ);      
        
        dΩ = Measure(get_triangulation(Vₘₛ.Uh), 4);
        L²Error[i] = sqrt(sum( ∫((Uₘₛʰ - Uex)*(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫((Uex)*(Uex))dΩ ));
        H¹Error[i] = sqrt(sum( ∫(A*∇(Uₘₛʰ - Uex)⊙∇(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫(A*∇(Uex)⊙∇(Uex))dΩ ))
        
        println("Done nc = "*string(nc))
      end
      Plots.plot!(plt1, 1 ./N, L²Error, label="(p="*string(p)*"), L² (l="*string(l)*")", lw=2)
      Plots.plot!(plt2, 1 ./N, H¹Error, label="(p="*string(p)*"), Energy (l="*string(l)*")", lw=2)
      Plots.scatter!(plt1, 1 ./N, L²Error, label="", markersize=2, xaxis=:log10, yaxis=:log10)
      Plots.scatter!(plt2, 1 ./N, H¹Error, label="", markersize=2, legend=:best, xaxis=:log10, yaxis=:log10)
      println("Done l = "*string(l))
    end
    Plots.plot!(plt1, 1 ./N, (1 ./N).^(p+3), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);
    Plots.plot!(plt2, 1 ./N, (1 ./N).^(p+2), label="Order "*string(p+2), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10);
    ##### ##### ##### ##### ##### ##### ##### ##### 
    # Script to visualize the basis functions
    ##### ##### ##### ##### ##### ##### ##### ##### 
    Λ = basis_vec_ms[:,(p+1)^2];
    Φ = FEFunction(Vₘₛ.Uh, Λ);
    writevtk(get_triangulation(Φ), "./2d_HigherOrderMS/basis_ms", cellfields=["u(x)"=>Φ]);
    writevtk(model_coarse, "./2d_HigherOrderMS/model");
  end 
end