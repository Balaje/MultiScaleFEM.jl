##### ###### ###### ###### ###### ###### ###### ###### ###### ####### ####### ####### ####### #
# Contains the interfaces to build the coarse scale fespaces and the saddle point matrices
##### ###### ###### ###### ###### ###### ###### ###### ###### ####### ####### ####### ####### #

function build_patch_coarse_spaces(model::DiscreteModel, p::Int64)
  ref_space = ReferenceFE(lagrangian, Float64, p)
  FESpace(model, ref_space, conformity=:L2)
end
function build_patch_fine_spaces(model::DiscreteModel, q::Int64)
  ref_space = ReferenceFE(lagrangian, Float64, q)
  FESpace(model, ref_space, conformity=:H1, dirichlet_tags="boundary")
end

function assemble_stima(fine_space::FESpace, A::Function, qorder::Int64)
  Ω = get_triangulation(fine_space)
  dΩ = Measure(Ω, qorder)
  a(u,v) = ∫( A*∇(u) ⊙ ∇(v) )dΩ
  assemble_matrix(a, fine_space, fine_space)
end