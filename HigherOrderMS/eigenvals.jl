####  ####  ####  #### ####  ####  ####  ####  ####  ####  ####  
# Script to test the eigenvalues of the operator.
####  ####  ####  #### ####  ####  ####  ####  ####  ####  ####  
include("HigherOrderMS.jl");
include("corrected_basis.jl");

gr() 

domain = (0.0,1.0)
Neps = 2^2
nds_micro = LinRange(domain[1], domain[2], Neps+1)
diffusion_micro = 0.05 .+ 1.95*rand(Neps+1)
function _D(x::T, nds_micro::AbstractVector{T}, diffusion_micro::Vector{T}) where T<:Number
  n = size(nds_micro, 1)
  for i=1:n
    if(nds_micro[i] < x < nds_micro[i+1])      
      return diffusion_micro[i+1]
    elseif(x==nds_micro[i])
      return diffusion_micro[i+1]
    elseif(x==nds_micro[i+1])
      return diffusion_micro[i+1]
    else
      continue
    end 
  end
end
A(x; nds_micro = nds_micro, diffusion_micro = diffusion_micro) = _D(x[1], nds_micro, diffusion_micro)

nf = 2^12
q = 1
qorder = 6

# Solve the fine scale problem onfce for exact solution
fine_scale_space = FineScaleSpace(domain, q, qorder, nf)
nds_fine = LinRange(domain[1], domain[2], q*nf+1)
stima = assemble_stiffness_matrix(fine_scale_space, A)
massma = assemble_mass_matrix(fine_scale_space, x->1.0)

p′s = [0,1,2,3]
plts =  Vector{Plots.Plot}(undef, length(p′s));
N = 2 .^(0:7);
l = N[end]

p = 3
evs = Matrix{AbstractVector{ComplexF64}}(undef, length(N), length(p′s))
i = 1
let
for p′ = p′s
  for nc = N
    # Obtain the map between the coarse and fine scale
    patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
    global basis_vec_ms₁ = compute_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices);
  
    # Compute the multiscale basis
    patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p′));
    global basis_vec_ms₂ = compute_l2_orthogonal_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices, p′; ntimes=1);      
  
    α = basis_vec_ms₁
    β = basis_vec_ms₂
    # println("nc = $nc, norm(α) = $( norm(α,Inf) ), norm(β) = $( norm(β,Inf) )");
  
    # Assemble the stiffness, mass matrices
    Kₘₛ = basis_vec_ms₁'*stima*basis_vec_ms₁; Mₘₛ = basis_vec_ms₁'*massma*basis_vec_ms₁; 
    Kₘₛ′ = basis_vec_ms₂'*stima*basis_vec_ms₂; Mₘₛ′ = basis_vec_ms₂'*massma*basis_vec_ms₂; 
    Lₘₛ = basis_vec_ms₂'*massma*basis_vec_ms₁
    Pₘₛ = basis_vec_ms₂'*stima*basis_vec_ms₁
  
    global 𝐌 = [Mₘₛ′ Lₘₛ; 
                Lₘₛ'  Mₘₛ];
    global 𝐊 = [Kₘₛ′ Pₘₛ; 
                Pₘₛ' Kₘₛ]
  
    global sM = 𝐌 |> collect 
    global sK = 𝐊 |> collect

    evs[i] = eigvals(sM\sK);    
    global i+=1
    println("Done nc = $nc")
  end  
  println("Done p′ = $p′")
end
end

## Plot the eigenvalue for p=0
plt_0 = Vector{Plots.Plot}(undef, length(N));
plt_1 = Vector{Plots.Plot}(undef, length(N));
plt_2 = Vector{Plots.Plot}(undef, length(N));
plt_3 = Vector{Plots.Plot}(undef, length(N));

for n=1:lastindex(N)
  plt_0[n] = Plots.scatter(real(evs[n,1]), imag(evs[n,1]), label="N = $(N[n]), p = $p, q = $(p′s[1])")
  plt_1[n] = Plots.scatter(real(evs[n,2]), imag(evs[n,2]), label="N = $(N[n]), p = $p, q = $(p′s[2])")
  plt_2[n] = Plots.scatter(real(evs[n,3]), imag(evs[n,3]), label="N = $(N[n]), p = $p, q = $(p′s[3])")
  plt_3[n] = Plots.scatter(real(evs[n,4]), imag(evs[n,4]), label="N = $(N[n]), p = $p, q = $(p′s[4])")
  vline!(plt_0[n], [0], lc=:black, label="x=0"); hline!(plt_0[n], [0], lc=:black, label="y=0")
  vline!(plt_1[n], [0], lc=:black, label="x=0"); hline!(plt_1[n], [0], lc=:black, label="y=0")
  vline!(plt_2[n], [0], lc=:black, label="x=0"); hline!(plt_2[n], [0], lc=:black, label="y=0")
  vline!(plt_3[n], [0], lc=:black, label="x=0"); hline!(plt_3[n], [0], lc=:black, label="y=0")
end

plts[p+1] = Plots.plot(plt_3..., layout=(4,2), size=(900,1200))