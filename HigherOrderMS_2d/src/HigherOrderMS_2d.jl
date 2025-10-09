__precompile__()

module HigherOrderMS_2d

include("./CoarseToFine.jl")
include("./Assemblers.jl")
include("./MultiscaleBases.jl")
include("./Stabilization.jl")

using HigherOrderMS_2d.CoarseToFine: coarsen, get_fine_nodes_in_coarse_elems

using HigherOrderMS_2d.Assemblers: assemble_loadvec, assemble_massma, assemble_stima, Λₖ
using HigherOrderMS_2d.Assemblers: assemble_ms_matrix, assemble_ms_loadvec

using HigherOrderMS_2d.MultiscaleBases: CoarseTriangulation, FineTriangulation, MultiScaleTriangulation, MultiScaleFESpace, lazy_fill
using HigherOrderMS_2d.MultiscaleBases: get_coarse_scale_patch_fine_scale_interior_node_indices, get_coarse_scale_patch_fine_scale_boundary_node_indices
using HigherOrderMS_2d.MultiscaleBases: get_coarse_scale_patch_coarse_elem_ids, get_coarse_scale_elem_fine_scale_node_indices
using HigherOrderMS_2d.MultiscaleBases: get_patch_coarse_elem
using HigherOrderMS_2d.MultiscaleBases: assemble_rect_matrix, assemble_lm_l2_matrix
using HigherOrderMS_2d.MultiscaleBases: MultiScaleCorrections, get_basis_functions, build_basis_functions!
using HigherOrderMS_2d.MultiscaleBases: \

using HigherOrderMS_2d.Stabilization: StabilizedMultiScaleFESpace, get_basis_functions


# Export all modules
export coarsen, get_fine_nodes_in_coarse_elems

export assemble_loadvec, assemble_massma, assemble_stima, assemble_rect_matrix, Λₖ
export assemble_ms_matrix, assemble_ms_loadvec


export CoarseTriangulation, FineTriangulation, MultiScaleTriangulation, MultiScaleFESpace, lazy_fill #, legendre_basis_on_fine_scale
export get_coarse_scale_patch_fine_scale_interior_node_indices, get_coarse_scale_patch_fine_scale_boundary_node_indices
export get_coarse_scale_patch_coarse_elem_ids, get_coarse_scale_elem_fine_scale_node_indices
export assemble_rect_matrix, assemble_lm_l2_matrix
export MultiScaleCorrections,get_basis_functions, build_basis_functions!
export \

export StabilizedMultiScaleFESpace


end # module HigherOrderMS_2d
