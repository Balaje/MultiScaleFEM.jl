include("HigherOrderMS.jl");

D(x) = (2 + cos(2π*x[1]/2e-2))^-1
f(x) = 1.0

nc = 2^8
nf = 2^16
q = 1
p = 1
l = 10
qorder = 2

patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));

# Compute MS bases
Kf, basis_vec_ms, U = compute_ms_basis((0.0,1.0), D, f, (q,p), (nf,nc), l, patch_indices_to_global_indices, qorder);
stima, massma, loadvec = Kf

# Solve the problem
# (-) Greedy sort the multiscale basis according to the element support
basis_elem_ms = broadcast(getindex, Fill(basis_vec_ms,nc), coarse_indices_to_fine_indices, ms_elem); 
basis_elem_ms_t = broadcast(transpose, basis_elem_ms);
# (-) Get the multiscale stiffness matrix
mat_el = BroadcastVector(mat_contribs, Fill(stima, nc), coarse_indices_to_fine_indices, 1:nc, nc);
ms_elem_mats = BroadcastVector(*, basis_elem_ms_t, mat_el, basis_elem_ms);
stima_ms = assemble_ms_matrix(ms_elem_mats, ms_elem, nc, p);
# (-) Get the multiscale load vector  
vec_el = BroadcastVector(vec_contribs, Fill(loadvec, nc), coarse_indices_to_fine_indices, 1:nc, nc);
ms_elem_vecs = BroadcastVector(*, basis_elem_ms_t, vec_el);
loadvec_ms = assemble_ms_vector(ms_elem_vecs, ms_elem, nc, p);
# (-) Solve the problem
sol = materialize(stima_ms)\materialize(loadvec_ms)
sol_fine_scale = get_solution(sol, basis_vec_ms);
plt = plot(LinRange(0,1,q*nf+1), sol_fine_scale);

@btime begin
  vec_el = BroadcastVector(vec_contribs, Fill($loadvec, $nc), $coarse_indices_to_fine_indices, $(1:nc), $nc);
  ms_elem_vecs = BroadcastVector(*, $basis_elem_ms_t, vec_el);
  loadvec_ms = assemble_ms_vector(ms_elem_vecs, $ms_elem, $nc, $p)
  materialize(loadvec_ms);
end;