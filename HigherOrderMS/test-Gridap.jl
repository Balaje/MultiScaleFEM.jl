using Gridap

# Legendre polynomials
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
function Bₖ(x,(l,p),nds)
    # nds is the coordinates of the element
    a,b=nds
    x̂ = -(a+b)/(b-a) + 2/(b-a)*x[1]
    (a ≤ x[1] ≤ b) ? Λₖᵖ(x̂,p)[l] : 0.0
end
#ε = 2^-3
#A(x) = (2 + cos(2π*x[1]/ε))^-1
A(x) = 1.0

domain = (0,1)
partition = (2^9,)
model = CartesianDiscreteModel(domain, partition)
Ω = Triangulation(model)
dΩ = Measure(Ω,20)

# H¹ subspace
order₁ = 2
reffe₁ = ReferenceFE(lagrangian, Float64, order₁)
U = TestFESpace(model, reffe₁, conformity=:H1, dirichlet_tags="boundary")
U₀ = TrialFESpace(U,x->0*x[1])

# L² subspace
order₂ = 2
reffe₂ = ReferenceFE(lagrangian, Float64, order₂)
V = TestFESpace(model, reffe₂, conformity=:L2)
V₀ = TrialFESpace(V)

X = MultiFieldFESpace([U,V])
X₀ = MultiFieldFESpace([U₀,V₀])

# Source function
R = (0.2,0.5)
Λₖ(y) = Bₖ(y,(order₂+1,order₂),R)

# Bilinear forms
a(u,v) = ∫( A*∇(u)⊙∇(v) )dΩ
b(λ,v) = ∫( λ*v )dΩ
lₕ(v) = ∫( Λₖ*v )dΩ
ã((Λ,λ), (v,μ)) = a(Λ,v) + b(λ,v) + b(μ,Λ) - 1e-20*b(λ,μ)
l̃((v,μ)) = lₕ(μ)

# Solve the problem
op = AffineFEOperator(ã, l̃, X₀, X)
Λ,λ = solve(op)

xvals = LinRange(domain[1], domain[2]-1e-10, partition[1])
#xvals = LinRange(0.44,0.46,100)
fxvals = map(x-> Λ(Point(x)), xvals)
plt = plot(xvals, fxvals, label="Λ(x):  H¹ Projection")

# Compute the L² projection
aₖ(ΠΛ,v) = ∫(ΠΛ*v)dΩ
lₖ(v) = ∫(Λ*v)dΩ
op1 = AffineFEOperator(b,lₖ,V,V)
ΠΛ = solve(op1)
xvals1 = R[1]:0.01:R[end]
fxvals1 = map(x-> ΠΛ(Point(x)), xvals1)
plt1 = plot(xvals1, fxvals1, label="ΠᵖΛ(x):  L² Projection of Λ")
plot!(plt1, xvals1, Λₖ.(xvals1), label="Λₖ(x): Legendre Polynomial")
xlims!(plt1, (xvals1[1],xvals1[end]))

@show norm(Λₖ.(xvals1) .- ΠΛ.(Point.(xvals1)), Inf)

plot(plt,plt1,layout=(2,1))