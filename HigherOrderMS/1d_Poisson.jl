include("./src/HigherOrderMS.jl");

#=
Problem data
=#
domain = (0.0,1.0)
D(x) = (1.0 + 0.5*cos(2π*x[1]/2^-6))^-1 # Oscillatory diffusion
f(x) = π^2*sin(π*x[1])

# Spatial discretization parameters
(length(ARGS)==4) && begin (nf, nc, p, l) = parse.(Int64, ARGS) end
if(length(ARGS)==0)
  nf = 2^11;
  p = 1;
  nc = 2^3;  
  l = 5; 
end

# Use Gridap to construct the space
fine_scale_space = FineScaleSpace(domain, 1, 6, nf)

# Solve the full problem once
stima = assemble_stiffness_matrix(fine_scale_space, D)
loadvec = assemble_load_vector(fine_scale_space, f)
fullnodes = 1:nf+1;
bnodes = [1, nf+1];
freenodes = setdiff(fullnodes, bnodes);
sol_ϵ = (stima[freenodes,freenodes])\(loadvec[freenodes]);
U = fine_scale_space.U
uₕ = FEFunction(U, [0.0; sol_ϵ; 0.0])


# Compute the map between coarse and fine scale
patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (1,p));
# Compute MS bases
basis_vec_ms = compute_ms_basis(fine_scale_space, D, p, nc, l, patch_indices_to_global_indices)
isStab = true
(isStab) && begin println("Stabilization on ..."); println(""); end
if(nc > 1 && isStab)
  γ = Cˡιₖ(fine_scale_space, D, p, nc, l);
  basis_vec_ms[:, 1:(p+1):(p+1)*nc] = γ;
end      

# Solve the problem
Kₘₛ = basis_vec_ms'*stima*basis_vec_ms;
Fₘₛ = basis_vec_ms'*loadvec;
sol2 = Kₘₛ\Fₘₛ;
sol_fine_scale = basis_vec_ms*sol2

# Compute the errors
dΩ = Measure(get_triangulation(U), 6)
uₘₛ = FEFunction(U, sol_fine_scale)    
e = uₕ - uₘₛ
L²Error = sqrt(sum(∫(e*e)dΩ));
H¹Error = sqrt(sum(∫(D*∇(e)⋅∇(e))dΩ));

println("")
println("$p \t $nc \t $l \t $L²Error \t $H¹Error")