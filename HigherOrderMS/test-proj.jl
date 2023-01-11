using Gridap

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
    (a < x[1] < b) ? Λₖᵖ(x̂,p)[l] : 0.0    
end

A(x) = 1/10

domain = (0,1)
partition = (10^6,)
model = CartesianDiscreteModel(domain, partition)
Ω = Triangulation(model)
dΩ = Measure(Ω,20)

# H¹ subspace
order₁ = 1
reffe₁ = ReferenceFE(lagrangian, Float64, order₁)
U = TestFESpace(model, reffe₁, conformity=:H1, dirichlet_tags="boundary")
U₀ = TrialFESpace(U,x->0*x[1])

Λₖ(y) = Bₖ(y,(1,3),(0.5,0.6))

# Bilinear forms
a(u,v) = ∫( A*∇(u)⊙∇(v) )dΩ
lₕ(v) = ∫( Λₖ*v )dΩ
op = AffineFEOperator(a,lₕ,U₀,U)
uh = solve(op)

xvals = domain[1]:0.01:domain[2]
fxvals = map(x->uh(Point(x)), xvals)
plt1 = plot(xvals,fxvals)
plt2 = plot(xvals, Λₖ.(xvals))#
plot(plt1,plt2,layout=(2,1))