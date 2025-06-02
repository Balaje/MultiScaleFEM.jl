##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Program to compute the multiscale bases and save them to the disk
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

include("./fileIO.jl");

# Accepts the project name and the element index as a command line argument
project_dir, project_name, cell_index = ARGS;
cell_index = parse(Int64, cell_index);
param_filename = project_dir*"/"*project_name*"/$(project_name)_params.csv";
domain, nf, nc, p, l, ntimes, vals_epsilon, tf, Δt, T₁ = read_problem_parameters(param_filename);

# Background fine scale discretization
FineScale = FineTriangulation(domain, nf);
reffe = ReferenceFE(lagrangian, T₁, 1);
V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1;vector_type=Vector{T₁});

A = CellField(vec(vals_epsilon), FineScale.trian)

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);

# Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);

# Assemble the fine scale matrices
K = assemble_stima(V₀, A, 4; T=T₁);
M = assemble_massma(V₀, x->1.0, 4; T=T₁);
L = assemble_rect_matrix(Ωₘₛ, p);
Λ = assemble_lm_l2_matrix(Ωₘₛ, p);

# Multiscale Space without stabilization
# γₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ))

# Multiscale Space with the stabilization
Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ));
γₘₛ = StabilizedMultiScaleFESpace(Vₘₛ, p, V₀, (K, L, Λ), domain, A);

# Multiscale Additional Corrections for the heat equation
Wₘₛ = Vector{MultiScaleCorrections}(undef, ntimes)
Wₘₛ[1] = MultiScaleCorrections(γₘₛ, p, (K, L, M, L));
for j=2:ntimes
  Wₘₛ[j] = MultiScaleCorrections(Wₘₛ[j-1], p, (K, L, M, L));
end

#### #### #### #### #### #### #### #### #### ####
# Compute and write the basis function
#### #### #### #### #### #### #### #### #### ####
filename = project_dir*"/"*project_name*"/$(project_name)_ms_basis_$(nc)$(p)$(l)_"*string(cell_index)*".csv"
B1 = γₘₛ.basis_vec_ms[cell_index];
write_basis_functions(B1, filename);

filename = Vector{String}(undef, ntimes)
for j=1:ntimes
    filename[j] = project_dir*"/"*project_name*"/$(project_name)_ms_basis_$(nc)$(p)$(l)_correction_level_$(j)_"*string(cell_index)*".csv"
    Bj = Wₘₛ[j].basis_vec_ms[cell_index];
    write_basis_functions(Bj, filename[j]);
end
