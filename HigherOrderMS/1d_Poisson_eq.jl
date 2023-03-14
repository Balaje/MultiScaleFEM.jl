include("HigherOrderMS.jl");


import Gridap.Arrays: evaluate!
struct *ᵐ <: Map end
evaluate!(cache,::*ᵐ,x,y) = x*y

D(x) = (2 + cos(2π*x[1]/2e-2))^-1
f(x) = 1.0

nc = 2^1
nf = 2^6
q = 1
p = 1
l = 1
qorder = 2

patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));

# Compute MS bases
Kf, Bs, U = compute_ms_basis((0.0,1.0), D, f, (q,p), (nf,nc), l, patch_indices_to_global_indices, qorder, coarse_indices_to_fine_indices, ms_elem);
stima, massma, loadvec = Kf
basis_vec_ms, B, Bt = Bs

# Solve the problem
# (-) Get the multiscale stiffness matrix
stima_el = lazy_map(mat_contribs, Fill(stima, nc), coarse_indices_to_fine_indices, 1:nc, Fill(nc,nc));
ms_elem_mats = lazy_map(*, SparseMatrixCSC{Float64,Int64}, Bt, stima_el, B);
stima_ms = assemble_ms_matrix(ms_elem_mats, ms_elem, nc, p)
# (-) Get the multiscale load vector  
loadvec_el = lazy_map(vec_contribs, Vector{Float64}, Fill(loadvec, nc), coarse_indices_to_fine_indices, 1:nc, Fill(nc,nc));
ms_elem_vecs = lazy_map(*ᵐ(), Vector{Float64}, Bt, loadvec_el);
loadvec_ms = assemble_ms_vector(ms_elem_vecs, ms_elem, nc, p);
# (-) Solve the problem
sol = stima_ms\collect(loadvec_ms)
sol_fine_scale = get_solution(sol, basis_vec_ms);
plt = plot(LinRange(0,1,q*nf+1), sol_fine_scale);