######### ############ ############ ############ ###########
# Compute the rate of convergence of the multiscale method
######### ############ ############ ############ ###########
using Plots
using BenchmarkTools
using NearestNeighbors
using SparseArrays
using LinearAlgebra
using ForwardDiff
using FastGaussQuadrature

include("basis_functions.jl")
include("assemble_matrices.jl")


#=
Problem data
=#
D(x) = @. 1.0
f(x) = @. π^2*sin(π*x)
u(x) = @. sin(π*x)
# ε = 2^-5
# D(x) = @. (2 + cos(2π*x/ε))^(-1)
# f(x) = @. 1.0
# u(x) = @. (x - x^2 + ε*(1/(4π)*sin(2π*x/ε) - 1/(2π)*x*sin(2π*x/ε) - ε/(4π^2)*cos(2π*x/ε) + ε/(4π^2)))
domain = (0,1)

"""
Function to compute the l2 and energy errors
"""
function error(bc, ue::Function, uh::AbstractVector{Float64}, nds::AbstractVector{Float64},
  elem::AbstractMatrix{Int64}, tree::KDTree, KDTrees::Vector{KDTree}, 
  quad::Tuple{Vector{Float64},Vector{Float64}}; Nfine=200)
  ∇ue(x) = ForwardDiff.derivative(ue,x)
  qs, ws = quad
  l2err = 0.0; h1err = 0.0
  nc = size(elem,1)
  for t=1:nc, i=1:lastindex(qs)
    cs = view(nds,view(elem,t,:))
    hlocal = (cs[2]-cs[1])/Nfine
    xlocal = cs[1]:hlocal:cs[2]
    for j=1:lastindex(xlocal)-1
      x̂ = (xlocal[j+1]-xlocal[j])*0.5 + (xlocal[j+1]-xlocal[j])*0.5*qs[i]
      l2err += ws[i]*(ue(x̂) - uₘₛ(bc, x̂, uh, tree, KDTrees))^2*(hlocal)*0.5
      h1err += ws[i]*D(x̂)*(∇ue(x̂) - ∇uₘₛ(bc, x̂, uh, tree, KDTrees))^2*(hlocal)*0.5
    end
  end
  sqrt(l2err), sqrt(h1err)
end

#=
Constant paramters
=#
p = 1; q = 1;
nf = 2^10
qorder = 2
quad = gausslegendre(qorder)

𝒩 = 2:16
L²Error = zeros(Float64,size(𝒩))
H¹Error = zeros(Float64,size(𝒩))

plt = plot()
plt1 = plot()
plt2 = plot()

for l in [4,5,6]
  for (nc,itr) in zip(𝒩,1:lastindex(𝒩))
    #=
    Precompute all the caches. Essential for computing the solution quickly
    =#
    npatch = min(2l+1,nc)
    # Construct the coarse mesh
    H = (domain[2]-domain[1])/nc
    nds = domain[1]:H:domain[2]
    tree = KDTree(nds')
    assem_L²L² = ([(p+1)*t+ti-p for _=0:p, ti=0:p, t=1:nc], 
    [(p+1)*t+tj-p for tj=0:p, _=0:p, t=1:nc], 
    [(p+1)*t+ti-p for ti=0:p, t=1:nc])
    assem_H¹H¹ = ([(q)*t+ti-(q-1) for _=0:q, ti=0:q, t=1:nf], 
    [(q)*t+tj-(q-1) for tj=0:q, _=0:q, t=1:nf], 
    [(q)*t+ti-(q-1) for ti=0:q, t=1:nf])          
    assem_H¹L² = ([(q)*Q+qᵢ-(q-1) for qᵢ=0:q, _=0:p, P=1:npatch, Q=1:nf], 
    [(p+1)*P+pᵢ-p for _=0:q, pᵢ=0:p, P=1:npatch, Q=1:nf], 
    [(p+1)*P+pᵢ-p for pᵢ=0:p, P=1:npatch])    
    # Connectivity of the coarse and fine domains
    elem_coarse = [i+j for i=1:nc, j=0:1]
    elem_fine = [i+j for i=1:nf, j=0:1]
    # Store only the non-zero entries of the matrices of the saddle point problems
    sKe = zeros(Float64,q+1,q+1,nf)
    sLe = zeros(Float64,q+1,p+1,npatch,nf)
    sFe = zeros(Float64,p+1,npatch)
    # Store the data for solving the multiscale problems
    KDTrees = Vector{KDTree}(undef,nc)
    MS_Basis = Array{Float64}(undef,q*nf+1,nc,p+1)
    # Some precomputed/preallocated data for the multiscale problem
    ndofs = npatch*(p+1)
    elem_ms = [
    begin 
      if(i < l+1)
        j+1
      elseif(i > nc-l)
        (ndofs-((npatch-1)*(p+1)))*(nc-(npatch-1))+(j)-(ndofs-1-((npatch-1)*(p+1)))
      else
        (ndofs-(2l*(p+1)))*(i-l)+j-(ndofs-1-(2l*(p+1)))
      end
    end  
    for i=1:nc,j=0:ndofs-1]
    assem_MS_MS = ([elem_ms[t,ti] for  t=1:nc, ti=1:ndofs, _=1:ndofs], 
    [elem_ms[t,tj] for  t=1:nc, _=1:ndofs, tj=1:ndofs], 
    [elem_ms[t,ti] for t=1:nc, ti=1:ndofs])
    sKms = zeros(Float64,nc,ndofs,ndofs)
    sFms = zeros(Float64,nc,ndofs)
    bc = basis_cache(elem_fine, q)
    local_basis_vecs = zeros(Float64, q*nf+1, ndofs)
    binds_1 = zeros(Int64, ndofs)
    cache = KDTrees, MS_Basis, local_basis_vecs, bc, binds_1
    
    
    #=
    Efficiently compute the solution to the saddle point problems.
    =#
    for i=1:nc
      start = max(1,i-l)
      last = min(nc,i+l)
      # Get the patch domain and connectivity
      patch_elem = elem_coarse[start:last,:] 
      patch = (nds[minimum(patch_elem)], nds[maximum(patch_elem)])  
      patch_elem = patch_elem .- (minimum(patch_elem) - 1)
      # Build the FEM nodes
      local h = (patch[2]-patch[1])/nf
      nds_fine = patch[1]:h:patch[2]
      nds_coarse = patch[1]:H:patch[2]
      # Build some data structures
      KDTrees[i] = KDTree(nds_fine') # KDTree for searching the points
      cache_q = basis_cache(q)
      cache_p = Vector{Float64}(undef,p+1)  
      # Fill up the matrices
      fillsKe!(sKe, cache_q, nds_fine, elem_fine, q, quad)
      fillsLe!(sLe, (cache_q,cache_p), nds_fine, nds_coarse, elem_fine, patch_elem, (q,p), quad)
      # Basis function of Vₕᵖ(K)
      function fₖ(x::Float64, j::Int64)
        res = Vector{Float64}(undef,p+1)
        nodes = [nds[elem_coarse[i,1]], nds[elem_coarse[i,2]]]
        Λₖ!(res, x, nodes, p)
        res[j]
      end
      # Compute new the basis functions
      for j=1:p+1
        # Fill up the vector
        fillsFe!(sFe, cache_p, nds_coarse, patch_elem, p, quad, y->fₖ(y,j))
        # Assemble the matrices
        KK = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe))
        LL = sparse(vec(assem_H¹L²[1]), vec(assem_H¹L²[2]), vec(sLe))
        FF = collect(sparsevec(vec(assem_H¹L²[3]), vec(sFe)))
        # Apply the boundary conditions
        tn = 1:q*nf+1
        bn = [1,q*nf+1]
        fn = setdiff(tn,bn)
        K = KK[fn,fn]; L = LL[fn,:]; F = FF
        # Solve the problem
        LHS = [K L; L' spzeros(Float64,size(L,2),size(L,2))]
        zcols = size(LHS,1)-rank(LHS)
        LHS = LHS[1:end-zcols,1:end-zcols]   
        dropzeros!(LHS)
        RHS = vcat(zeros(Float64,size(K,1)),F);
        RHS = RHS[1:end-zcols]
        RHS = LHS\RHS
        Λ = vcat(0, RHS[1:size(K,1)], 0)
        copyto!(view(MS_Basis,:,i,j), Λ)  
      end
    end
    
    #=
      Now solve the multiscale problems.
        The multiscale basis are stored in the variable MS_Basis
    =#
    # Compute the local and global stiffness matrices
    fillsKms!(sKms, cache, elem_coarse, nds, p, l, quad; Nfine=nf)
    fillsFms!(sFms, cache, elem_coarse, nds, p, l, quad, f; Nfine=nf)
    # Assemble and the global system
    Kₘₛ = sparse(vec(assem_MS_MS[1]), vec(assem_MS_MS[2]), vec(sKms))
    Fₘₛ = collect(sparsevec(vec(assem_MS_MS[3]), vec(sFms)))
    sol = (Kₘₛ\Fₘₛ)
    # Compute the error in the solution
    bc = basis_cache(elem_coarse, elem_fine, p, q, l, MS_Basis)
    L²Error[itr], H¹Error[itr] = error(bc, u, sol, nds, elem_coarse, tree, KDTrees, quad; Nfine=nf)

    println("Done nc = "*string(nc))
  end

  println("Done l = "*string(l))
  plot!(plt, 1 ./𝒩, L²Error, label="L² (l="*string(l)*")", xaxis=:log10, yaxis=:log10, lw=2)
  plot!(plt1, 1 ./𝒩, H¹Error, label="Energy (l="*string(l)*")", lw=2)
  scatter!(plt, 1 ./𝒩, L²Error, label="", markersize=2)
  scatter!(plt1, 1 ./𝒩, H¹Error, label="", markersize=2, legend=:outertopright)
end

plot!(plt1, 1 ./𝒩, (1 ./𝒩).^2, label="Order 2", ls=:dash, lc=:black)
plot!(plt, 1 ./𝒩, (1 ./𝒩).^3, label="Order 3", ls=:dash, lc=:black)

plot!(plt2, 0:0.01:1, u.(0:0.01:1), label="Exact", lw=1, lc=:black)