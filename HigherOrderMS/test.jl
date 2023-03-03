using FastGaussQuadrature
using LinearAlgebra
using Plots

include("preallocation.jl")
include("basis-functions.jl")
include("assemble_matrices.jl")

nc = 2
nf = 2^16
q = 1
p = 1

prob_data = PreAllocateMatrices.preallocate_matrices((0.0,1.0), nc, nf, 2, (q,p))
nds_fine = prob_data[1][3]
elem_fine = prob_data[1][4]
quad = gausslegendre(4)

cache = AssembleMatrices.assembler_cache(nds_fine, elem_fine, quad, 1)

stima = AssembleMatrices.assemble_matrix!(cache, x->1.0, MultiScaleBases.∇φᵢ!, MultiScaleBases.∇φᵢ!, -1)
fn = 2:q*nf
K = lu(stima[fn,fn])
solvec = Vector{Float64}(undef, length(fn))
fill!(solvec,0.0)
for i=1:1000
  loadvec = AssembleMatrices.assemble_vector!(cache, x->1.0,  MultiScaleBases.φᵢ!, 1)      
  f = loadvec[fn]
  fill!(solvec,0.0)
  ldiv!(solvec, K, f)
  (i%1000 == 0) && print("Done 1000\n")
end
# plt = plot(nds_fine[fn], solvec)
# plot!(plt, nds_fine[fn], 0.5*nds_fine[fn].*(1 .- nds_fine[fn]))
# display(plt)