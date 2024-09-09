include("2d_HigherOrderMS.jl")

# Problem description
domain = (0.0, 1.0, 0.0, 1.0)
A(x) = 1.0
f(x) = 2π^2*sin(π*x[1])*sin(π*x[2]);

# Construct the triangulation of the fine-scale
nf = 2^5
model = CartesianDiscreteModel(domain, (nf,nf))
reffe = ReferenceFE(lagrangian, Float64, 1)
V0 = TestFESpace(model,reffe,conformity=:H1, dirichlet_tags="boundary")
Ω_fine = Triangulation(model)
# Construct the triangulation of the coarse-scale
nc = 2^1;
p = 3;
model_coarse = CartesianDiscreteModel(domain, (nc,nc))
Ω_coarse = Triangulation(model_coarse)

# Obtain the coarse-to-fine map
nsteps =  (Int64(log2(nf/nc)))
coarse_to_fine_map = coarsen(model, nsteps); 

l = 1;
Ωₘₛ = MultiScaleTriangulation(domain, nf, nc, l);

D = CellField(A, Ωₘₛ.Ωf)
Vₘₛ = MultiScaleFESpace(Ωₘₛ, 1, p, D, 4);