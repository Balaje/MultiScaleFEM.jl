######### ############ ############ ############ ############ ###
# Program to compare direct method vs the new multiscale method #
######### ############ ############ ############ ############ ###

include("meshes.jl")
include("assemblers.jl")
include("fespaces.jl")
include("basis_functions.jl")
include("local_matrix_vector.jl")
include("assemble_matrices.jl")

ε = 2^-5
A(x) = @. (2 + cos(2π*x/ε))^(-1)
u(x) = @. (x - x^2 + ε*(1/(4π)*sin(2π*x/ε) - 1/(2π)*x*sin(2π*x/ε) - ε/(4π^2)*cos(2π*x/ε) + ε/(4π^2)))
f(x) = @. 1.0
qorder = 3

#= 
Direct method using piecewise elements
=#
p = 1 # Polynomial order
n = 2^5 # Number of elements
Ω = 𝒯((0,1),n) # Build the triangulation
Vₕ = H¹Conforming(Ω, 1, [0,n*p+1]) # Build the finite element space
Kₐ = MatrixAssembler(H¹ConformingSpace(), p, Ω.elems) # Matrix Assembler for H¹-innerproduct
Fₐ = VectorAssembler(H¹ConformingSpace(), p, Ω.elems) # Vector Assembler for H¹-innerproduct
MM,KK = assemble_matrix(Vₕ, Kₐ, A, x->1.0; qorder=qorder) # Assemble the FE-matrix
FF = assemble_vector(Vₕ, Fₐ, f; qorder=qorder) # Assemble the FE-vector
K = KK[2:n*p,2:n*p] # Matrix after applying the boundary conditions
F = FF[2:n*p] # Vector after applying 0 boundary conditions
uh = K\F; # Solve the Poisson problem
uh = vcat(0.0, uh, 0.0) # Append the boundary values
plt1 = plot(Ω.nds, uh, label="Direct FEM", lw=2, lc=:red) # Plot the solution
@show "Direct method done."
#=
Multiscale method
=#
p = 1 # Polynomial order of coarse space
q = 1 # Polynomial order of fine space
n = 2 # Number of elements in the coarse space
nₚ = 2^9 # Number of elements in the fine space
l = 5 # Patch size parameter
Ω = 𝒯((0,1),n) # Triangulation
Vₕᴹˢ = MultiScale(Ω, A, (q,p), l, [1,n*p+1]; Nfine=nₚ, qorder=qorder) # Multiscale FE-space
MSₐ = MatrixAssembler(MultiScaleSpace(), p, Ω.elems, l) # Matrix assembler for MS-innerproduct
MSₗ = VectorAssembler(MultiScaleSpace(), p, Ω.elems, l) # Vector assembler for MS-innerproduct
Mₘₛ,Kₘₛ = assemble_matrix(Vₕᴹˢ, MSₐ, A, x->1.0; qorder=qorder, Nfine=40) # Assemble the MSFE-matrix
Fₘₛ = assemble_vector(Vₕᴹˢ, MSₗ, f; qorder=qorder, Nfine=40) # Assemble the MSFE-vector
uh = Kₘₛ\Fₘₛ # Solve the Poisson problem
xvals = LinRange(0,1,400)
uhxvals =  [uₘₛ(x, uh, Vₕᴹˢ) for x in xvals] # Compute the nodal solutions
plot!(plt1, xvals, uhxvals, lc=:blue, lw=2, label="MS Method")
@show "MS-FE method done"
#=
Exact solution
=#
plot!(plt1, xvals, u.(xvals), lc=:black, lw=1, ls=:dash, label="Exact sol")