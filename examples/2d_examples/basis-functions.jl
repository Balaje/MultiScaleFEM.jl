###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 

using Pkg
Pkg.activate(".")

using Gridap
using MultiscaleFEM
using SparseArrays
using ProgressMeter
using SplitApplyCombine

using MPI
comm = MPI.COMM_WORLD
MPI.Init()
mpi_rank = MPI.Comm_rank(comm)

# include("./time-dependent.jl")

domain = (0.0, 1.0, 0.0, 1.0);

# Fine scale space description
nf = 2^7;
nc = 2^4;
p = 3;
l = 32; # Patch size parameter

# A(x) = (0.5 + 0.5*cos(2π/2^-5*x[1])*cos(2π/2^-5*x[2]))^-1
# Background fine scale discretization
FineScale = FineTriangulation(domain, nf);
reffe = ReferenceFE(lagrangian, Float64, 1);
V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1);
epsilon = 2^5
repeat_dims = (Int64(nf/epsilon), Int64(nf/epsilon))
a₁,b₁ = (0.5,1.5)
if(mpi_rank==0)
  rand_vals = rand(epsilon^2);
else
  rand_vals = zeros(epsilon^2);
end
MPI.Bcast!(rand_vals, 0, comm)
vals_epsilon = repeat(reshape(a₁ .+ (b₁-a₁)*rand_vals, (epsilon, epsilon)), inner=repeat_dims)
A = CellField(vec(vals_epsilon), FineScale.trian)
K = assemble_stima(V₀, A, 4);
M = assemble_massma(V₀, x->1.0, 4);

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);

# Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);
L = assemble_rect_matrix(Ωₘₛ, p);
Λ = assemble_lm_l2_matrix(Ωₘₛ, p);

# Multiscale space of order p
Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ));
basis_vec_ms = Vₘₛ.basis_vec_ms;

# Correction space of order 0 Wₘₛ ⟂ₐ Wₚ
q = 0;
L₀ = assemble_rect_matrix(Ωₘₛ, q);
Λ₀ = assemble_lm_l2_matrix(Ωₘₛ, q);
Vₘₛ′ = MultiScaleFESpace(Ωₘₛ, q, V₀, (K, L₀, Λ₀));
Wₘₛ =  MultiScaleCorrections(Vₘₛ′, p, (K, L, M, L₀));


num_coarse_cells = num_cells(CoarseScale.trian);
num_fine_cells = num_cells(FineScale.trian);
Zb = get_coarse_scale_patch_fine_scale_boundary_node_indices(Ωₘₛ);
Zi = get_coarse_scale_patch_fine_scale_interior_node_indices(Ωₘₛ);
Zpatch = MultiscaleFEM.MultiscaleBases.get_patch_coarse_elem.(Ref(CoarseScale.trian), Ref(CoarseScale.tree), Ref(l), 1:num_coarse_cells);
coarse_coords = get_cell_coordinates(CoarseScale.trian);
fine_coords = FineScale.trian.grid.node_coords;

# Visualize the patch/element data
function plot_patch_data!(plt, Zb, Zi, Zpatch, coarse_coords, fine_coords, el, l, alpha)
  Plots.scatter!(plt, Tuple.(fine_coords[Zi[el]]), ms=0.75, label="", msw=0.0, ma=alpha)
  Plots.scatter!(plt, Tuple.(fine_coords[Zb[el]]), ms=1.5 , label="", msw=0.0, ma=alpha)
  Plots.scatter!(plt, Tuple.(combinedims(coarse_coords[Zpatch[el]])) |> vec, label="", msw=0.0, size=(600,600), ma=alpha)
  Plots.scatter!(plt, Tuple.(coarse_coords[el]) |> vec, label="Element $el, l=$l", msw=0.0, size=(600,600), ma=alpha)
  # nc = length(coarse_coords) |> sqrt |> Int64
  # Plots.hline!(plt, LinRange(0,1,nc+1), alpha=0.5)
  # Plots.vline!(plt, LinRange(0,1,nc+1), alpha=0.5)
  Plots.xlims!((-0.1,1.1))
  Plots.ylims!((-0.1,1.1))
  plt
end

# Visualize the basis functions
function plot_basis_function!(plt1, V, el, j, p)
  @assert 1 ≤ j ≤ (p+1)^2
  i = (el-1)*(p+1)^2+j
  b = get_basis_functions(V)[el][:,i];  
  nf = size(b,1) |> sqrt |> Int64
  # nc = sqrt(size(get_basis_functions(V),1)) |> Int64  
  Plots.contour!(plt1, LinRange(0,1,nf), LinRange(0,1,nf), reshape(b, (nf,nf))', rightmargin=1.5Plots.cm)
  # Plots.hline!(plt1, LinRange(0,1,nc+1), alpha=0.5)
  # Plots.vline!(plt1, LinRange(0,1,nc+1), alpha=0.5)
  Plots.xlims!((-0.1,1.1))
  Plots.ylims!((-0.1,1.1))
  plt1
end

##### ##### ###### ##### ###### ##### ###### 
# Plotting handles. Uncomment for plotting.
##### ##### ###### ##### ###### ##### ###### 
using Plots
plt = Plots.plot();
plt1 = Plots.plot();
plt2 = Plots.plot();
plt3 = Plots.plot();
el = 1
jp = 1
jq = (q+1)^2
# el = 10
plot_patch_data!(plt, Zb, Zi, Zpatch, coarse_coords, fine_coords, el, l, 1)
# Plot the multiscale basis and corrections
plot_patch_data!(plt1, Zb, Zi, Zpatch, coarse_coords, fine_coords, el, l, 0.5)
plot_basis_function!(plt1, Vₘₛ, el, jp, p);
plot_patch_data!(plt2, Zb, Zi, Zpatch, coarse_coords, fine_coords, el, l, 0.5)
plot_basis_function!(plt2, Vₘₛ′, el, jq, q);
plot_patch_data!(plt3, Zb, Zi, Zpatch, coarse_coords, fine_coords, el, l, 0.5)
plot_basis_function!(plt3, Wₘₛ, el, jq, q);

B = spzeros(Float64, length(fine_coords), num_coarse_cells*(p+1)^2)
B₁ = spzeros(Float64, length(fine_coords), num_coarse_cells*(q+1)^2)

build_basis_functions!((B,B₁), (Vₘₛ,Wₘₛ), comm);

if(mpi_rank==0)
  println("a-Orthogonality a(Vₘₛ,Wₘₛ) = $(norm(B'*K*B₁)) ≈ 0.0");
  println("a-Orthogonality a(Vₘₛ,Vₘₛ) = $(norm(B'*K*B)) ≂̸ 0.0");
  println("L²-Orthogonality (Vₘₛ,Wₘₛ) = $(norm(B'*M*B₁)) ≂̸ 0.0 ");
end