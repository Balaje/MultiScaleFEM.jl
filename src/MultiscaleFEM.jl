module MultiscaleFEM

include("./CoarseToFine.jl")
include("./Assemblers.jl")
include("./MultiscaleBases.jl")

using MultiscaleFEM.CoarseToFine: coarsen, get_fine_nodes_in_coarse_elems

using MultiscaleFEM.Assemblers: assemble_loadvec, assemble_massma, assemble_stima, Λₖ
using MultiscaleFEM.Assemblers: assemble_ms_matrix, assemble_ms_loadvec

using MultiscaleFEM.MultiscaleBases: CoarseTriangulation, FineTriangulation, MultiScaleTriangulation, MultiScaleFESpace, lazy_fill #, legendre_basis_on_fine_scale
using MultiscaleFEM.MultiscaleBases: get_coarse_scale_patch_fine_scale_interior_node_indices, get_coarse_scale_patch_fine_scale_boundary_node_indices
using MultiscaleFEM.MultiscaleBases: get_coarse_scale_patch_coarse_elem_ids, get_coarse_scale_elem_fine_scale_node_indices
using MultiscaleFEM.MultiscaleBases: assemble_rect_matrix, assemble_lm_l2_matrix
using MultiscaleFEM.MultiscaleBases: MultiScaleCorrections, get_basis_functions, build_basis_functions!


# Export all modules
export coarsen, get_fine_nodes_in_coarse_elems

export assemble_loadvec, assemble_massma, assemble_stima, assemble_rect_matrix, Λₖ
export assemble_ms_matrix, assemble_ms_loadvec


export CoarseTriangulation, FineTriangulation, MultiScaleTriangulation, MultiScaleFESpace, lazy_fill #, legendre_basis_on_fine_scale
export get_coarse_scale_patch_fine_scale_interior_node_indices, get_coarse_scale_patch_fine_scale_boundary_node_indices
export get_coarse_scale_patch_coarse_elem_ids, get_coarse_scale_elem_fine_scale_node_indices
export assemble_rect_matrix, assemble_lm_l2_matrix
export MultiScaleCorrections,get_basis_functions, build_basis_functions!


end # module MultiscaleFEM
