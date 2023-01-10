###############################################################################
# Functions to compute the multi-scale basis by solving the localized problem.
###############################################################################
using Gridap
"""
mutable struct Rˡₕ <: Any
Λ::FEFunction
λ::FEFunction
Ω::Triangulation
end
"""
mutable struct Rˡₕ <: Any
    Λ::FEFunction
    λ::FEFunction
    Ω::Triangulation
end
"""
Function to solve the local problem on Vₕᵖ(Nˡ(K)). We input:
    (1) Λₖ : The Lagrange basis function on k
    (2) A : The Diffusion coefficient
    (3) xn : Patch of order l (Should be supplied as an interval externally)
    (4) kwargs: 
        1) fespace = (q,p). Order of polynomials where q:Fine Space and p:The multiscale method.
        2) N = Number of points in the fine space.
        3) qorder = Quadrature order for the fine-scale problem.
"""
function Rˡₕ(Λₖ::Function, A::Function, xn::Tuple; fespace=(1,1), N=50, qorder=10)
    q,p=fespace   
    # Solve the problem using Gridap
    domain = xn
    partition = (N,)
    model = CartesianDiscreteModel(domain, partition)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,qorder)
    # H¹ subspace    
    rq = ReferenceFE(lagrangian, Float64, q)
    U = TestFESpace(model, rq, conformity=:H1, dirichlet_tags="boundary")
    U₀ = TrialFESpace(U, x->0*x[1])
    # L² subspace
    rp = ReferenceFE(lagrangian, Float64, p)
    V = TestFESpace(model, rp, conformity=:L2)
    V₀ = TrialFESpace(V)
    # Multifield space
    X = MultiFieldFESpace([U,V])
    X₀ = MultiFieldFESpace([U₀,V₀])
    # Bilinear forms
    a(u,v) = ∫( A*∇(u)⊙∇(v) )dΩ
    b(λ,v) = ∫( λ*v )dΩ
    lₕ(v) = ∫( Λₖ*v )dΩ
    ã((Λ,λ), (v,μ)) = a(Λ,v) + b(λ,v) + b(μ,Λ) 
    l̃((v,μ)) = lₕ(μ)
    # Solve the problem
    op = AffineFEOperator(ã, l̃, X₀, X)
    Λ,λ = solve(op)
    Rˡₕ(Λ,λ,Ω)
end
Base.show(io::IO, z::Rˡₕ) = print(io, "Local basis Rˡₕ on [",z.Ω.grid.node_coords[1],",",z.Ω.grid.node_coords[end],"], ")