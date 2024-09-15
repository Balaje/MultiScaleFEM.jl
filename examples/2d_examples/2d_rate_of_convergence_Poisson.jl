using Gridap
using MultiscaleFEM
using Plots
using ProgressBars

# Problem description
domain = (0.0, 1.0, 0.0, 1.0)
A(x) = 1.0
# # A(x) = (2 + 0.5*cos(2π/2^-5*x[1])*cos(2π/2^-5*x[2]))^-1
f(x) = 2π^2*sin(π*x[1])*sin(π*x[2])

# # Construct the triangulation of the fine-scale
nf = 2^8
model = CartesianDiscreteModel(domain, (nf,nf))
Ω_fine = Triangulation(model)
# reffe = ReferenceFE(lagrangian, Float64, 1)
# Vh = TestFESpace(Ω_fine, reffe, conformity=:H1, dirichlet_tags="boundary");
# Vh0 = TrialFESpace(Vh, 0.0);
# dΩ_fine = Measure(Ω_fine, 5);
# a(u,v) = ∫(A*(∇(v)⊙∇(u)))dΩ_fine;
# b(v) = ∫(v*f)dΩ_fine;
# op = AffineFEOperator(a,b,Vh0,Vh);
# Uex = solve(op);

N = [2,4,8,16];
L²Error = zeros(Float64, length(N));
H¹Error = zeros(Float64, length(N));

plt1 = Plots.plot();
plt2 = Plots.plot();

let 
  FineScale = FineTriangulation(domain, nf);
  reffe = ReferenceFE(lagrangian, Float64, 1);
  V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1); 
  K = assemble_stima(V₀, A, 0);
  F = assemble_loadvec(V₀, f, 3);
  M = assemble_massma(V₀, x->1.0, 6);
  for p=[2]
    for l=[p+2,p+3]
      for (nc,i) = zip(N, 1:length(N))
        # Coase scale discretization
        CoarseScale = CoarseTriangulation(domain, nc, l);

        # Multiscale Triangulation
        Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);
        L = assemble_rect_matrix(Ωₘₛ, V₀, p);
        Λ = assemble_lm_l2_matrix(Ωₘₛ, p);
        
        Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ));
        basis_vec_ms, patch_interior_fine_scale_dofs, coarse_dofs  = Vₘₛ.basis_vec_ms;
        Ks, Ls, Λs = Vₘₛ.fine_scale_system;
        # Lazy fill the fine-scale vector
        Fs = lazy_fill(F, num_cells(CoarseScale.trian));
        
        B = zero(L);
        for i=ProgressBar(1:nc^2)
          B[patch_interior_fine_scale_dofs[i], coarse_dofs[i]] = basis_vec_ms[i];  
        end
        
        # # Multiscale Stiffness and RHS
        Kₘₛ = assemble_ms_matrix(B, K);
        Fₘₛ = assemble_ms_loadvec(B, F);
        solₘₛ = Kₘₛ\Fₘₛ;
        Uₘₛ = B*solₘₛ;
        Uₘₛʰ = FEFunction(Vₘₛ.Uh, Uₘₛ);          
        
        
        Uex = CellField(x->sin(π*x[1])*sin(π*x[2]), FineScale.trian);
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
    # ##### ##### ##### ##### ##### ##### ##### ##### 
    # # Script to visualize the basis functions
    # ##### ##### ##### ##### ##### ##### ##### ##### 
    # Λ = basis_vec_ms[:,(p+1)^2];
    # Φ = FEFunction(Vₘₛ.Uh, Λ);
    # writevtk(get_triangulation(Φ), "basis_ms", cellfields=["u(x)"=>Φ]);
    # writevtk(model_coarse, "model");
  end 
end