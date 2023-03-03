using FastGaussQuadrature
using LinearAlgebra
using Plots

include("preallocation.jl")
include("basis-functions.jl")
include("assemble_matrices.jl")
include("solve.jl")

nc = 2
nf = 2^6
q = 2
p = 1

prob_data = PreAllocateMatrices.preallocate_matrices((0.0,1.0), nc, nf, 2, (q,p))
nds_fine = prob_data[1][3]
elem_fine = prob_data[1][4]
quad = gausslegendre(4)

cache = AssembleMatrices.assembler_cache(nds_fine, elem_fine, quad, q)
stima = AssembleMatrices.assemble_matrix!(cache, x->1.0, MultiScaleBases.∇φᵢ!, MultiScaleBases.∇φᵢ!, -1) # Have to do it only once in most cases
loadvec = AssembleMatrices.assemble_vector!(cache, x->1.0,  MultiScaleBases.φᵢ!, 1)  # Efficient assembly

fn = 2:q*nf

cache = SolveLinearSystem.solution_cache(stima, fn)
SolveLinearSystem.solve!(cache, loadvec, fn)
solvec = SolveLinearSystem.get_solution(cache, [0,0])

plt = plot(nds_fine, solvec[1:q:q*nf+1])
plot!(plt, nds_fine, 0.5*nds_fine.*(1 .- nds_fine))