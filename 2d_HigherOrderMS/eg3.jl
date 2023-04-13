###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 
include("2d_HigherOrderMS.jl")

domain = (0.0, 1.0, 0.0, 1.0)

# Fine scale space description
nf = 2^2
q = 1
nc = 2^0
p = 1
ms_space = MultiScaleFESpace(domain, q, p, nf, nc)

num_coarse_cells = num_cells(get_triangulation(ms_space.UH))
num_fine_cells = num_cells(get_triangulation(ms_space.Uh))

coarse_to_fine_map = get_coarse_to_fine_map(num_coarse_cells, num_fine_cells)