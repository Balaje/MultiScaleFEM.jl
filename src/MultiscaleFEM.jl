module MultiscaleFEM

include("./CoarseToFine.jl")
include("./Assemblers.jl")
include("./MultiscaleBases.jl")

using MultiscaleFEM.CoarseToFine: coarsen, get_fine_nodes_in_coarse_elems
using MultiscaleFEM.Assemblers: assemble_loadvec, assemble_massma, assemble_stima, assemble_rect_matrix, assemble_rhs_matrix, Λₖ, saddle_point_system
using MultiscaleFEM.MultiscaleBases: MultiScaleTriangulation, MultiScaleFESpace

export coarsen, get_fine_nodes_in_coarse_elems
export assemble_loadvec, assemble_massma, assemble_stima, assemble_rect_matrix, assemble_rhs_matrix, Λₖ, saddle_point_system
export MultiScaleTriangulation, MultiScaleFESpace

end # module MultiscaleFEM
