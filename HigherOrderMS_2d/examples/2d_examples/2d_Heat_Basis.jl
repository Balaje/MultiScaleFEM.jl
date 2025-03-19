###### ######## ######## ######## ######## ######### 
# Program to test the multiscale basis computation #
###### ######## ######## ######## ######## ######### 

# # Run this the first time
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Quadmath
T₁ = Float128

using Gridap
using MultiscaleFEM
using SparseArrays
using ProgressMeter
using DelimitedFiles

include("./time-dependent.jl")

domain = T₁.((0.0, 1.0, 0.0, 1.0));

##### ##### ##### ##### ##### ##### ##### 
# Temporal discretization parameters
##### ##### ##### ##### ##### ##### ##### 
Δt = 2^-7
tf = 1.0
ntime = ceil(Int, tf/Δt)
BDF = 4

##### ##### ##### ##### ##### ##### ##### 
# Spatial discretization parameters
##### ##### ##### ##### ##### ##### ##### 
# @assert length(ARGS) > 0 "I need more info..."
if(length(ARGS)==7) 
  nf, nc, p, l, ntimes, PROBLEM_NAME, BASIS_IND = ARGS
  nf, nc, p, l, ntimes, BASIS_IND = parse.(Int64, (nf, nc, p, l, ntimes, BASIS_IND));
else
  nf = 2^4
  nc = 2^2
  p = 1
  l = 1
  ntimes = 1
  BASIS_IND = 1
  PROBLEM_NAME = "Hello"  
end

f(x,t) = T₁(2π^2*sin(π*x[1])*sin(π*x[2])*(sin(t)))
u₀(x) = T₁(0.0)

# Background fine scale discretization
FineScale = FineTriangulation(domain, nf);
reffe = ReferenceFE(lagrangian, T₁, 1);
V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1;vector_type=Vector{T₁});

# Random field
epsilon = min(2^5, nf)
repeat_dims = (Int64(nf/epsilon), Int64(nf/epsilon))
a₁,b₁ = T₁.((0.5,1.5))
rand_vals = ones(T₁,epsilon^2)
vals_epsilon = repeat(reshape(a₁ .+ (b₁-a₁)*rand_vals, (epsilon, epsilon)), inner=repeat_dims)

# vals_epsilon = readdlm("./coefficient.txt");
A = CellField(vec(vals_epsilon), FineScale.trian)

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);

# Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);

# Assemble the fine scale matrices
K = assemble_stima(V₀, A, 4; T=T₁);
M = assemble_massma(V₀, x->1.0, 4; T=T₁);
L = assemble_rect_matrix(Ωₘₛ, p);
Λ = assemble_lm_l2_matrix(Ωₘₛ, p);

# Multiscale Space without stabilization
# γₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ))

# # Multiscale Space with the stabilization
Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ));
γₘₛ = StabilizedMultiScaleFESpace(Vₘₛ, p, V₀, (K, L, Λ), domain, A);

# Multiscale Additional Corrections for the heat equation
Wₘₛ = Vector{MultiScaleCorrections}(undef, ntimes)
Wₘₛ[1] = MultiScaleCorrections(γₘₛ, p, (K, L, M, L)); 
for j=2:ntimes
  Wₘₛ[j] = MultiScaleCorrections(Wₘₛ[j-1], p, (K, L, M, L)); 
end

using DataFrames, CSV

function write_basis_functions(V, FILENAME)
  I, J, VALS = findnz(V)
  df = DataFrame((a=I, b=J, c=VALS))
  CSV.write(FILENAME, df)
end
function read_basis_functions(FILENAME, ::Type{T}, mat_size) where T<:Real
  B = CSV.read(FILENAME, DataFrame, types=[Int64, Int64, String]);
  sparse(B.a, B.b, parse.(T,B.c), mat_size...)
end

MS_BASIS_FILENAME = "./"*string(PROBLEM_NAME)*"/"*string(PROBLEM_NAME)*"_$(nc)$(p)$(l)_"*string(BASIS_IND)*".csv"
B1 = γₘₛ.basis_vec_ms[BASIS_IND];
write_basis_functions(B1, MS_BASIS_FILENAME);

MS_BASIS_CORRECTION_FILENAME = Vector{String}(undef, ntimes)
for j=1:ntimes
  MS_BASIS_CORRECTION_FILENAME[j] = "./"*string(PROBLEM_NAME)*"/"*string(PROBLEM_NAME)*"_$(nc)$(p)$(l)$(j)_"*string(BASIS_IND)*".csv"
  local A = Wₘₛ[j].basis_vec_ms[BASIS_IND];
  write_basis_functions(A, MS_BASIS_CORRECTION_FILENAME[j]);
end