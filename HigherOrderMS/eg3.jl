###########################################
# Program to solve the Multiscale problem.
###########################################

# include("eg1.jl"); # Contains the basis functions
include("meshes.jl")
include("assemblers.jl")
include("fespaces.jl")
include("basis_functions.jl")
include("local_matrix_vector.jl")
include("assemble_matrices.jl")

#ε = 2^-2
#A(x) = @. (2 + cos(2π*x/ε))^(-1)
#u(x) = @. (x - x^2 + ε*(1/(4π)*sin(2π*x/ε) - 1/(2π)*x*sin(2π*x/ε) - ε/(4π^2)*cos(2π*x/ε) + ε/(4π^2)))
f(x) = @. 1.0
A(x) = @. 0.5
u(x) = @. x*(1-x)

# Problem parameters
p = 2
q = 1
l = 2
n = 2^2
nₚ = 2^4
num_nei = 2
qorder = 2
# Discretize the domain
Ω = 𝒯((0,1),n)
# Build the Multiscale space. Contains the basis functions in the global sense
Vₕᴹˢ = MultiScale(Ω, A, (q,p), l, [1,n*p+n-p]; Nfine=nₚ, qorder=qorder)
# Plot the basis function
el = 2
Rₛ = Vₕᴹˢ.basis
plt2 = plot()
for k=1:p+1
  R = Rₛ[k,el]
  plot!(plt2, R.nds, R.Λ, lw=2, label="Basis "*string(k))
  xlims!(plt2,(0,1))
end
# Build the assembler
MSₐ = MatrixAssembler(MultiScaleSpace(), p, Ω.elems, l)
MSₗ = VectorAssembler(MultiScaleSpace(), p, Ω.elems, l)
# Compute the full stiffness and mass matrices
Mₘₛ,Kₘₛ = assemble_matrix(Vₕᴹˢ, MSₐ, A, x->1.0; qorder=qorder, Nfine=nₚ)
Fₘₛ = assemble_vector(Vₕᴹˢ, MSₗ, f; qorder=qorder, Nfine=nₚ)
#--
# Boundary conditions are applied into the basis functions
#--
uh = Kₘₛ\Fₘₛ
xvals = Ω.nds
uhxvals =  [uₘₛ(x, uh, Vₕᴹˢ) for x in collect(xvals)]
uxvals = u.(xvals)
plt = plot(xvals, uhxvals, label="Multiscale FEM")
plot!(plt, xvals, uxvals, label="Exact sol")