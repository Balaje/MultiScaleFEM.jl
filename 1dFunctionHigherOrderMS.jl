##################################################################
# Modules used to implement the Higher Order Multiscale Methods
##################################################################

include("1dFunctions.jl");

using ForwardDiff
using FastGaussQuadrature
using LinearAlgebra
using SparseArrays
"""
TODO:
1) Implement the Legendre Polynomials
    - Shift it to the elements in the support of K. It belongs to L²(Nˡ(K)).
    - Have a parameter for the degree of the Legendre polynomial.
2) Solve the local finite element problems
    - Divide the support of the element into smaller elements.
    - Solve the finite element problems in the small elements.
    - Gaussian quadrature needs to be high to resolve a(u,v).
3)
"""

"""
Function to generate the Legendre polynomials upto order p.
The polynomials are defined in the reference interval (-1,1)
"""
function Λₖᵖ(x, p::Int64)
  (p==0) && return [1.0]
  (p==1) && return [1.0, x]
  (p > 1) && begin
    res = Vector{Float64}(undef, p+1)
    fill!(res,0.)
    res[1] = 1.0
    res[2] = x[1]
    for j=2:p
      res[j+1] = (2j-1)/(j)*x*res[j] + (1-j)/(j)*res[j-1]
    end
    return res
  end
end

"""
Lagrange basis function for the fine scale problem.
  (Let us start with order 1 and 2 first.)
"""
function ϕ̂(x, p::Int64)
  xq = LinRange(-1,1,p+1)
  Q = [xq[i]^j for i=1:p+1, j=0:p]
  IdMatrix = I(p+1)
  A = Q\IdMatrix
  A'*[x^i for i=0:p]
end
function ∇ϕ̂(x, p::Int64)
  res = Vector{Float64}(undef,p+1)
  fill!(res,0.0)
  for i=1:p+1
    φᵢ(y) = ϕ̂(y, p)[i]
    res[i] = ForwardDiff.derivative(φᵢ, x)
  end
  res
end
"""
Basis function and gradient for the Lagrange multiplier.
  (This is the L² function and is equal to the
    Legendre polynomials)
"""
ψ̂(x, p) = Λₖᵖ(x, p)
function ∇ψ̂(x, p)
  (p==0) && return 0.0
  (p==1) && return [0.0, 1.0]
  (p > 1) && begin
    dRes = Vector{Float64}(undef, p+1)
    Res = Vector{Float64}(undef, p+1)
    fill!(dRes,0.); fill!(Res,0.)
    dRes[1] = 0.0;   dRes[2] = 1.0
    Res[1] = 1.0; Res[2] = x
    for j=2:p
      Res[j+1] = (2j-1)/(j)*x*Res[j] + (1-j)/(j)*Res[j-1]
      dRes[j+1] = (2j-1)/(j)*(Res[j] + x*dRes[j]) + (1-j)/(j)*dRes[j-1]
    end
    return dRes
  end
end

"""
We define the set of functions that carry out the construction of the
linear system for higher order polynomials.
"""
# Function to get the local matrix-vector system for any polynomial of order p
function _local_matrix_vector_H¹_H¹(xn, A::Function, f::Function, quad, h, p)
  Me = Array{Float64}(undef, p+1, p+1)
  Ke = Array{Float64}(undef, p+1, p+1)
  Fe = Vector{Float64}(undef, p+1)
  res = Vector{Float64}(undef, p+1)
  res1 = Vector{Float64}(undef, p+1)

  fill!(Me, 0.); fill!(Ke, 0.); fill!(Fe, 0.);
  fill!(res,0.); fill!(res1,0)

  qs,ws = quad
  J = 0.5*h
  for q=1:length(qs)
    x̂ = qs[q]
    x = (xn[2] + xn[1])*0.5 .+ 0.5*h*x̂
    res = ϕ̂(x̂,p)
    res1 = ∇ϕ̂(x̂,p)
    # Loop over the local matrices
    for i=1:p+1
      ϕᵢ = res[i]
      ∇ϕᵢ = res1[i]
      Fe[i] += ws[q]*( f(x)*ϕᵢ )*J
      for j=1:p+1
        ϕⱼ = res[j]
        ∇ϕⱼ = res1[j]
        Me[i,j] += ws[q]*( ϕᵢ * ϕⱼ )*J
        Ke[i,j] += ws[q]*( A(x) * ∇ϕᵢ * ∇ϕⱼ )*J^-1
      end
    end
  end
  Me, Ke, Fe, res, res1, p
end

# Function to get the local rectangular mass matrix for the local problem.
function _localmassmatrix_H¹_L²((p₁,p₂); h=2)
  qopt = p₂+2
  qs,ws = gausslegendre(qopt)
  m,n=p₁,p₂
  res = Matrix{Float64}(undef,m+1,n+1)
  fill!(res, 0.)
  for i=1:m+1, j=1:n+1, q=1:qopt
    res[i,j] += ws[q] * ϕ̂(qs[q],p₁)[i] * ψ̂(qs[q],p₂)[j]
  end
  res = res*(0.5*h)
end

function _localmassmatrix_L²_L²(p::Int64; h=2)
  qopt = p+2
  qs,ws = gausslegendre(qopt)
  m,n=p+1,p+1
  res = Matrix{Float64}(undef,m,n)
  fill!(res, 0.)
  for i=1:m, j=1:n, q=1:qopt
    res[i,j] += ws[q] * ψ̂(qs[q],p)[i] * ψ̂(qs[q],p)[j]
  end
  res = res*(0.5*h)
end

function _localvector_L²(xn, p::Int64, f::Function; qorder=10, h=2)
  quad = gausslegendre(qorder)
  qs, ws = quad
  qorder = length(qs)
  res = Vector{Float64}(undef,p+1)
  fill!(res,0.0)
  J = h*0.5
  for j=1:p+1
    for q=1:qorder
      x̂ = (xn[2]+xn[1])*0.5 .+ (h)*0.5*qs[q]
      res[j] += ws[q] * f(x̂) * ψ̂(qs[q],p)[j] * J
    end
  end
  res
end

# Function to obtain the elem arrays with
function new_connectivity_matrices(elem, (q,p))
  # Returns the H¹ (q-polynomial) and L² (p-polynomial)
  nel = size(elem,1)
  new_elem = Matrix{Int64}(undef, nel, q+1)
  new_elem_1 = Matrix{Int64}(undef, nel, p+1)
  for i=1:nel
    new_elem[i,:] = elem[i,1]+(i-1)*(q-1): elem[i,2]+i*(q-1)
    new_elem_1[i,:] = elem[i,1]+(i-1)*(p): elem[i,2]+i*(p)-1
  end
  new_elem, new_elem_1
end

# Function to get the assembler for any polynomial of order p
function get_assembler(elem, p)
  nel = size(elem,1)
  new_elem, _ = new_connectivity_matrices(elem, (p,p))
  iM = Array{Int64}(undef, nel, p+1, p+1)
  jM = Array{Int64}(undef, nel, p+1, p+1)
  iV = Array{Int64}(undef, nel, p+1)
  fill!(iM,0); fill!(jM, 0); fill!(iV,0)
  for t=1:nel
    for ti=1:p+1
      iV[t,ti] = new_elem[t,ti]
      for tj=1:p+1
        iM[t,ti,tj] = new_elem[t,ti]
        jM[t,ti,tj] = new_elem[t,tj]
      end
    end
  end
  iM, jM, iV
end

# Function to assemble the H¹(D) × H¹(D) matrix and vector
function assemble_matrix_H¹_H¹(ijk, nodes, els, A::Function, f::Function, p; qorder=10)
  i,j,k = ijk
  nel = size(i,1)
  hlocal = nodes[2]-nodes[1]
  quad = gausslegendre(qorder)

  sKe = Array{Float64}(undef, nel, p+1, p+1)
  sMe = Array{Float64}(undef, nel, p+1, p+1)
  sFe = Array{Float64}(undef, nel, p+1)

  fill!(sKe,0.0); fill!(sMe,0.0); fill!(sFe,0.0)

  # Do the assembly
  for t=1:nel
    #cs = view(nodes, view(i,t,:,1))
    cs = nodes[els[t,:],:]
    Me, Ke, Fe, res, res1, p = _local_matrix_vector_H¹_H¹(cs, A, f, quad, hlocal, p)
    for ti=1:p+1
      sFe[t,ti] = Fe[ti]
      for tj=1:p+1
        sMe[t,ti,tj] = Me[ti,tj]
        sKe[t,ti,tj] = Ke[ti,tj]
      end
    end
  end

  K = sparse(vec(i), vec(j), vec(sKe))
  M = sparse(vec(i), vec(j), vec(sMe))
  F = collect(sparsevec(vec(k), vec(sFe)))
  droptol!(M,1e-20), droptol!(K,1e-20), F
end


# Function to assemble the H¹(D) × L²(D) matrix
function assemble_matrix_H¹_L²(node, elem, (p₁,p₂))
  hlocal = node[2,1]-node[1,1]
  Mlocal = _localmassmatrix_H¹_L²((p₁,p₂); h=hlocal) #(p+1) × p matrix
  nel = size(elem,1)
  # new_elem = Matrix{Int64}(undef, nel, p₁+1)
  # new_elem_1 = Matrix{Int64}(undef, nel, p₂+1)
  # for i=1:nel
  #   new_elem[i,:] = elem[i,1]+(i-1)*(p₁-1): elem[i,2]+i*(p₁-1)
  #   new_elem_1[i,:] = elem[i,1]+(i-1)*(p₂): elem[i,2]+i*(p₂)-1
  # end
  new_elem, new_elem_1 = new_connectivity_matrices(elem, (p₁,p₂))
  # Size of the index vectors = (dim*p+1)*(dim*p)*nel
  ii = Vector{Int64}(undef, (p₁+1)*(p₂+1)*nel)
  jj = Vector{Int64}(undef, (p₁+1)*(p₂+1)*nel)
  sA = Vector{Float64}(undef, (p₁+1)*(p₂+1)*nel)
  fill!(ii,0)
  fill!(jj,0)
  fill!(sA,0.0)
  index = 0
  for i=1:p₁+1, j=1:p₂+1
    ii[index+1:index+nel] = new_elem[:,i]
    jj[index+1:index+nel] = new_elem_1[:,j]
    sA[index+1:index+nel] .= Mlocal[i,j]
    index = index+nel
  end
  droptol!(sparse(ii, jj, sA), 1e-20)
end

# Function to assemble the L²(D) × L²(D) matrix
function assemble_matrix_L²_L²(node, elem, p)
  hlocal = node[2,1]-node[1,1]
  Mlocal = _localmassmatrix_L²_L²(p; h=hlocal)
  nel = size(elem,1)
  # new_elem = Matrix{Int64}(undef, nel, p+1)
  # for i=1:nel
  #   new_elem[i,:] = elem[i,1]+(i-1)*(p): elem[i,2]+i*(p)-1
  # end
  _, new_elem = new_connectivity_matrices(elem, (p,p))
  ii = Vector{Int64}(undef, (p+1)*(p+1)*nel)
  jj = Vector{Int64}(undef, (p+1)*(p+1)*nel)
  sA = Vector{Float64}(undef, (p+1)*(p+1)*nel)
  fill!(ii,0)
  fill!(jj,0)
  fill!(sA,0.0)
  index = 0
  for i=1:p+1, j=1:p+1
    @inbounds begin
      ii[index+1:index+nel] = new_elem[:,i]
      jj[index+1:index+nel] = new_elem[:,j]
      sA[index+1:index+nel] .= Mlocal[i,j]
      index = index+nel
    end
  end
  droptol!(sparse(ii, jj, sA), 1e-20)
end

# Function to assemble the L²(D) vector
function assemble_vector_L²(node, elem, p, f::Function; qorder=10)
  hlocal = node[2,1] - node[1,1]
  nel = size(elem,1)
  res = Vector{Float64}(undef, nel*(p+1))
  fill!(res,0.)
  # new_elem = Matrix{Int64}(undef, nel, p+1)
  # for i=1:nel
  #   new_elem[i,:] = elem[i,1]+(i-1)*(p): elem[i,2]+i*(p)-1
  # end
  _, new_elem = new_connectivity_matrices(elem, (p,p))
  for i=1:nel
    cs = node[elem[i,:],:]
    Felocal = _localvector_L²(cs, p, f; qorder=qorder, h=hlocal)
    res[new_elem[i,1]:new_elem[i,end]] = Felocal
  end
  res
  res[abs.(res) .< 1e-20] .= 0
  res
end

"""
Function to solve the local problems to obtain the projection of the
Legendre basis in H¹₀(Nˡ(K)).
"""
mutable struct Rˡₕ <: Any
  nds
  els::Matrix{Int64}
  Λ⃗::Vector{Float64}
  λ⃗::Vector{Float64}
end

function Rˡₕ(Λₖ::Function, A::Function, xn::Tuple; fespace=(1,1), N=50, qorder=10)
  q,p=fespace
  # Now solve the problem
  nds, els = mesh(xn, N)
  hlocal = (xn[2]-xn[1])/N
  new_nodes = xn[1]:(hlocal)/q:xn[2]
  nel = size(els,1)
  # Boundary, Interior and Total Nodes
  tn = 1:size(new_nodes,1)
  bn = [1, length(new_nodes)]
  fn = setdiff(tn,bn)
  # Assemble the system
  assembler = get_assembler(els,q)
  ~, KK, ~ = assemble_matrix_H¹_H¹(assembler, nds, els, A, x-> 0*x, q; qorder=qorder)
  LL = assemble_matrix_H¹_L²(nds, els, (q,p))
  MM = assemble_matrix_L²_L²(nds, els, p)
  FF = assemble_vector_L²(nds, els, p, Λₖ; qorder=qorder)
  ## Apply the boundary conditions
  K = KK[fn,fn]; L = LL[fn,:]; Lᵀ = L'; M = MM; F = FF
  A = [K L; Lᵀ -1e-10*M]
  b = Vector{Float64}(undef, (p+1)*nel+length(fn))
  fill!(b,0.0)
  b[length(fn)+1:end] = F
  sol = A\b;
  Λ⃗ = sol[1:length(fn)]
  λ⃗ = sol[length(fn):end]
  Rˡₕ(nds,
      els,
      vcat(0,Λ⃗,0),
      λ⃗)
end

function Λ̃ₖˡ(x, R::Rˡₕ; fespace=(1,1))
  nds = R.nds; elem=R.els; uh = R.Λ⃗; λₕ = R.λ⃗
  q,p=fespace
  nel = size(elem,1)
  new_elem, _ = new_connectivity_matrices(elem,(q,p))
  for i=1:nel
    new_elem[i,:] = elem[i,1]+(i-1)*(q-1): elem[i,2]+i*(q-1)
    uh_elem = uh[new_elem[i,:]]
    nds_elem = nds[elem[i,:]]
    hl = nds_elem[2]-nds_elem[1]
    if(nds_elem[1] ≤ x ≤ nds_elem[2])
      x̂ = -(nds_elem[2]+nds_elem[1])/hl + (2/hl)*x
      return dot(uh_elem, ϕ̂(x̂,q))
    else
      continue
    end
  end
end
function ∇Λ̃ₖˡ(x, R::Rˡₕ; fespace=(1,1))
  nds = R.nds; elem=R.els; uh = R.Λ⃗; λₕ = R.λ⃗
  q,p=fespace
  nel = size(elem,1)
  new_elem, _ = new_connectivity_matrices(elem,(q,p))
  for i=1:nel
    new_elem[i,:] = elem[i,1]+(i-1)*(q-1): elem[i,2]+i*(q-1)
    uh_elem = uh[new_elem[i,:]]
    nds_elem = nds[elem[i,:]]
    hl = nds_elem[2]-nds_elem[1]
    if(nds_elem[1] ≤ x ≤ nds_elem[2])
      x̂ = -(nds_elem[2]+nds_elem[1])/hl + (2/hl)*x
      return dot(uh_elem, ∇ϕ̂(x̂,q))*(2/hl)
    else
      continue
    end
  end
end

# Basis of the L² subspace (Legendre polynomials)
function Bₖ(x,(l,p),nds)
  # nds is the coordinates of the element
  a,b=nds
  x̂ = -(a+b)/(b-a) + 2/(b-a)*x
  (a < x < b) ? ψ̂(x̂,p) : zeros(Float64,p+1)
end

# Function to obtain the new multiscale method connectivity matrix
# The connectivity depends on the support of the basis (parameter, l)
# (Overload the previous connectivity matrix function)
function new_connectivity_matrices(elems, (q,p), l)
  new_elem, _ = new_connectivity_matrices(elems, (q,p))
  new_elem_ms = Matrix{Int64}(undef,nel,(p+1)*(2l+1))
  fill!(new_elem_ms,0)
  for i=1:nel
    #start = (new_elem[i,1]-(p+1)*l ≤ 0) ? 1 : new_elem[i,1]-(p+1)*l
    #last = (new_elem[i,end]+(p+1)*l ≥ new_elem[nel,end]) ? new_elem[nel,end] : new_elem[i,end]+(p+1)*l
    #new_elem_ms[i,1:length(start:last)] =  start:last
    start = ((i-l) ≤ 0) ? 1 : (i-l)
    last = ((i+l) ≥ nel) ? nel : (i+l)
    elinds = vec(permutedims(new_elem[start:last,:],[2,1]))
    new_elem_ms[i,1:length(elinds)] = elinds
  end
  new_elem_ms
end

function _local_matrix_vector_MS_MS(xn, A::Function, f::Function, quad, H, fespace,
                                    dim, R::Vector{Rˡₕ})
  Ke = Array{Float64}(undef, dim, dim)
  Fe = Vector{Float64}(undef, dim)
  Me = Array{Float64}(undef, dim, dim)
  @assert dim == length(R)
  q,p=fespace
  fill!(Ke, 0.); fill!(Fe, 0.); fill!(Me,0.)
  qs, ws = quad
  J = 0.5*H
  for qp=1:length(qs)
    x̂ = qs[qp]
    x = (xn[2]+xn[1])*0.5 + 0.5*H*x̂
    for i=1:dim
      Fe[i] += ws[qp] * f(x) * Λ̃ₖˡ(x, R[i]; fespace=fespace) * J
      for j=1:dim
        Ke[i,j] += ws[qp] * A(x) * ∇Λ̃ₖˡ(x, R[i]; fespace=fespace)*
          ∇Λ̃ₖˡ(x, R[j]; fespace=fespace) * J
        Me[i,j] += ws[qp] * Λ̃ₖˡ(x, R[i]) * Λ̃ₖˡ(x, R[j]) * J
      end
    end
  end
  Ke, Me, Fe
end
function assemble_matrix_MS_MS(nodes, els, ms_elems, A::Function, f::Function, fespace; qorder=10, Nfine=100, plot_basis=1)
  q,p = fespace
  nel = size(els,1)
  hlocal = nodes[2]-nodes[1]
  quad  = gausslegendre(qorder)
  sKe = Array{Float64}(undef, nel, (2l+1)*(p+1), (2l+1)*(p+1))
  sMe = Array{Float64}(undef, nel, (2l+1)*(p+1), (2l+1)*(p+1))
  sFe = Array{Float64}(undef, nel, (2l+1)*(p+1))
  fill!(sKe,0.0); fill!(sMe,0.0); fill!(sFe,0.0)
  # Compute all the basis
  RˡₕΛₖ = Matrix{Rˡₕ}(undef,nel,p+1)
  for k=1:nel
    elcoords = (nodes[els[k,1]],nodes[els[k,2]])
    start = (k-l)>0 ? k-l : 1
    last = (k+l)<nel ? k+l : nel
    for i=1:p+1
      Λₖ(y) = Bₖ(y, (l,p), elcoords)[i]
      RˡₕΛₖ[k,i] = Rˡₕ(Λₖ, A, (nodes[els[start,1]], nodes[els[last,2]]);
                       fespace=fespace, N=Nfine, qorder=10)
    end
  end
  if(plot_basis > 0)
    plt = plot()
    for i=1:p+1
      xs = RˡₕΛₖ[plot_basis,i].nds                  
      fxs = map(x->Λ̃ₖˡ(x, RˡₕΛₖ[plot_basis,i]; fespace=fespace), xs)      
      plot!(plt, xs, fxs, lw=2, label="Basis "*string(i))
      xlims!(plt, (0,1))            
    end
    savefig(plt, "local_basis.pdf")
  end
  # Do the assembly
  K = spzeros(Float64, maximum(ms_elems), maximum(ms_elems))
  M = spzeros(Float64, maximum(ms_elems), maximum(ms_elems))
  F = Vector{Float64}(undef, maximum(ms_elems))

  # Let us use the naive way to assemble the matrices for now
  for t=1:nel
    cs = nodes[els[t,:],:]
    el = nonzeros(sparsevec(ms_elems[t,:]))
    start = (t-l)>0 ? t-l : 1
    last = (t+l)<nel ? t+l : nel
    R = RˡₕΛₖ[start:last,:]
    R = permutedims(R, [2,1])
    Ke,Me,Fe = _local_matrix_vector_MS_MS(cs, A, f, quad, hlocal, fespace,
                                          length(el), vec(R))
    for i in 1:length(el)
      F[el[i]] += Fe[i]
      for j in 1:length(el)
        K[el[i],el[j]] += Ke[i,j]
        M[el[i],el[j]] += Me[i,j]
      end
    end
  end
  droptol!(K, 1e-20), droptol!(M, 1e-20), F
end

# Solve the multiscale FEM problem
using Plots
#ε = 1
#A(x) = @. (2 + cos(2π*x/ε))^-1
A(x) = @. 1
# Define the coarse and fine space
nodes, elems = mesh((0,1), 10)
H = nodes[2]-nodes[1]
# Define some parameter
p = 2 # Order of the method i.e., degree of the L² subspace
q = 1 # Polynomial degree of the fine mesh
l = 1 # Size of the support of the new basis
fespace = (q,p) # FESpace pair (H¹, L²)
nel = size(elems,1)
new_nodes = nodes[1]:H/p:nodes[end]
tn = 1:length(new_nodes)
bn = [1, length(new_nodes)]
fn = setdiff(tn, bn)
ms_elem_matrix = new_connectivity_matrices(elems, (p,p), l)
#assem = new_assembler(ms_elem_matrix, p, l)
# Compute the projection of the L² functions onto the H¹ space
K,M,F = assemble_matrix_MS_MS(nodes, elems, ms_elem_matrix, A, x-> 1, fespace; qorder=10, Nfine=2^9, plot_basis=5)
#sol = K[fn,fn]\F[fn]
#plt = plot(new_nodes, vcat(0,sol,0))
