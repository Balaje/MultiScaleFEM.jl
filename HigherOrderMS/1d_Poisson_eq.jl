include("HigherOrderMS.jl");

domain = (0.0,1.0)

D(x) = (2 + cos(2π*x[1]/2e-2))^-1
f(x) = 1.0

nc = 2^1
nf = 2^16
q = 1
p = 1
l = 1
qorder = 2

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
plt3 = plot(LinRange(domain[1], domain[2], q*nf+1), sol_fine_scale)