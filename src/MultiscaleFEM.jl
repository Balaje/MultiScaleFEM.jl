module MultiscaleFEM

include("./CoarseToFine.jl")
include("./Assemblers.jl")
include("./MultiscaleBases.jl")

using MultiscaleFEM.CoarseToFine: coarsen, get_fine_nodes_in_coarse_elems

using MultiscaleFEM.Assemblers: assemble_loadvec, assemble_massma, assemble_stima, assemble_rect_matrix, assemble_rhs_matrix, Λₖ
using MultiscaleFEM.Assemblers: assemble_ms_matrix, assemble_ms_loadvec, assemble_fine_scale_from_ms, solve_ms_problem

using MultiscaleFEM.MultiscaleBases: CoarseTriangulation, FineTriangulation, MultiScaleTriangulation, MultiScaleFESpace, solve_schur_complement, lazy_fill
using MultiscaleFEM.MultiscaleBases: get_coarse_scale_patch_fine_scale_interior_node_indices, get_coarse_scale_patch_fine_scale_boundary_node_indices
using MultiscaleFEM.MultiscaleBases: get_coarse_scale_patch_coarse_elem_ids, get_coarse_scale_elem_fine_scale_node_indices


# Export all modules
export coarsen, get_fine_nodes_in_coarse_elems

export assemble_loadvec, assemble_massma, assemble_stima, assemble_rect_matrix, assemble_rhs_matrix, Λₖ
export assemble_ms_matrix, assemble_ms_loadvec, assemble_fine_scale_from_ms, solve_ms_problem


export CoarseTriangulation, FineTriangulation, MultiScaleTriangulation, MultiScaleFESpace, solve_schur_complement, lazy_fill
export get_coarse_scale_patch_fine_scale_interior_node_indices, get_coarse_scale_patch_fine_scale_boundary_node_indices
export get_coarse_scale_patch_coarse_elem_ids, get_coarse_scale_elem_fine_scale_node_indices


end # module MultiscaleFEM
