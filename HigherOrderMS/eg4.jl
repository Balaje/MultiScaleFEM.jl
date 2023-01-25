#############################################
# Program to compute the rate of convergence
#############################################

include("meshes.jl")
include("assemblers.jl")
include("fespaces.jl")
include("basis_functions.jl")
include("local_matrix_vector.jl")
include("assemble_matrices.jl")

## Function to compute the energy and L2 norm
function error(uh::Vector{Float64}, V::T,  u::Function, A::Function; qorder=5, Nfine=20) where T<:MultiScale
  ∇u(x) = ForwardDiff.derivative(u,x)
  nodes = V.trian.nds
  els = V.trian.elems
  new_els = V.new_elem
  basis = V.basis
  err_L2 = 0; err_energy = 0
  nel = size(els,1)
  qs,ws = gausslegendre(qorder)  
  for t=1:nel    
    cs = nodes[els[t,:],:]
    b_inds = new_els[t,:]
    hlocal = cs[2]-cs[1]
    xlocal = LinRange(cs[1],cs[2],Nfine)
    uh_elem = uh[b_inds]
    for i=1:lastindex(xlocal)-1
      for k=1:lastindex(qs)
        x = (xlocal[i+1]+xlocal[i])*0.5 + 0.5*(xlocal[i+1]-xlocal[i])*qs[k]
        ϕᵢ = [Λ̃ˡₚ(x, basis[i], basis[i].U; num_neighbours=2) for i in b_inds]
        ∇ϕᵢ = [∇Λ̃ˡₚ(x, basis[i], basis[i].U; num_neighbours=2) for i in b_inds]
        err_L2 += ws[k]*(dot(uh_elem, ϕᵢ)-u(x))^2*(xlocal[i+1]-xlocal[i])*0.5
        err_energy += ws[k]*A(x)*(dot(uh_elem, ∇ϕᵢ)-∇u(x))^2*(xlocal[i+1]-xlocal[i])*0.5
      end    
    end    
  end
  err_L2, err_energy
end

### Solve a problem and compute the error
#=
=#
# ε = 2^-2
# A(x) = @. (2 + cos(2π*x/ε))^(-1)
# u(x) = @. (x - x^2 + ε*(1/(4π)*sin(2π*x/ε) - 1/(2π)*x*sin(2π*x/ε) - ε/(4π^2)*cos(2π*x/ε) + ε/(4π^2)))
# f(x) = @. 1.0
# A(x) = @. 0.5
# u(x) = @. x*(1-x)
f(x) = @. π^2*sin(π*x)
A(x) = @. 1.0
u(x) = @. sin(π*x)

  # Problem parameters
p = 1;  q = 1;  
nₚ = 2^9; # Fine mesh partition.  
qorder = 3; # Quadrature rule
N = [1,2,4,8,16,32]; # Coarse mesh partition

plt1 = plot(); plt2 = plot()
for l in [1,2,3,4,5]
  @show l
  L²errs = Vector{Float64}(undef,length(N))
  H¹errs = Vector{Float64}(undef,length(N))
  for i in 1:lastindex(N)
    local Ω = 𝒯((0,1),N[i]) # Computational domain
    # Assembling strategy
    local MSₐ = MatrixAssembler(MultiScaleSpace(), p, Ω.elems, l)
    local MSₗ = VectorAssembler(MultiScaleSpace(), p, Ω.elems, l)
    local Vₕᴹˢ = MultiScale(Ω, A, (q,p), l, [1,n*p+n]; Nfine=nₚ, qorder=qorder) # Multiscale space
    # Stiffness matrix and load vector
    local _,Kₘₛ = assemble_matrix(Vₕᴹˢ, MSₐ, A, x->1.0; qorder=qorder, Nfine=100)
    local Fₘₛ = assemble_vector(Vₕᴹˢ, MSₗ, f; qorder=qorder, Nfine=100)
    # Solve the problem
    local uh = Kₘₛ\Fₘₛ
    local L²,H¹ = error(uh, Vₕᴹˢ, u, A; qorder=qorder, Nfine=100)
    @show L², H¹
    L²errs[i] = L²; H¹errs[i] = H¹;
  end
  plot!(plt1, (1 ./N), (H¹errs), yaxis=:log10, xaxis=:log10, label="H¹ (l="*string(l)*")", lw=2)
  plot!(plt2, (1 ./N), (L²errs), yaxis=:log10, xaxis=:log10, label="L² (l="*string(l)*")", lw=2)
end
plot!(plt1, (1 ./N), (1 ./N).^2, yaxis=:log10, xaxis=:log10, label="order 2", ls=:dash, lc=:black)
plot!(plt2, (1 ./N), (1 ./N).^3, yaxis=:log10, xaxis=:log10, label="order 3", ls=:dash, lc=:black)

plot!(plt1, (1 ./N), (1 ./N).^3, yaxis=:log10, xaxis=:log10, label="order 3", ls=:dash, lc=:black)
plot!(plt2, (1 ./N), (1 ./N).^4, yaxis=:log10, xaxis=:log10, label="order 4", ls=:dash, lc=:black)

plot!(plt1, (1 ./N), (1 ./N).^4, yaxis=:log10, xaxis=:log10, label="order 4", ls=:dash, lc=:black)
plot!(plt2, (1 ./N), (1 ./N).^5, yaxis=:log10, xaxis=:log10, label="order 5", ls=:dash, lc=:black)

plt3 = plot(plt1,plt2,layout=(1,2))