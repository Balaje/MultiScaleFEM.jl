###### ######## ######## ######## ######## ######### ######### ######### #########
# Program to set the problem parameters and the fine scale discretization
###### ######## ######## ######## ######## ######### ######### ######### #########

include("./fileIO.jl")
include("./time-dependent.jl")

T₁ = Float64
domain = T₁.((0.0, 1.0, 0.0, 1.0));

##### ##### ##### ##### ##### ##### #####
# Temporal discretization parameters
##### ##### ##### ##### ##### ##### #####
Δt = 2^-8
tf = 1.0
ntime = ceil(Int, tf/Δt)
BDF = 4

##### ##### ##### ##### ##### ##### #####
# Spatial discretization parameters
##### ##### ##### ##### ##### ##### #####
if(length(ARGS)==7)
    nf, nc, p, l, ntimes, project_dir, project_name = ARGS
    nf, nc, p, l, ntimes = parse.(Int64, (nf, nc, p, l,
                                          ntimes));
else
    nf = 2^7
    nc = 2^1
    p = 3
    l = 6
    ntimes = 2
    PROJECT_NAME = "Test"
end

# Random field
epsilon = min(2^6, nf)
repeat_dims = (Int64(nf/epsilon), Int64(nf/epsilon))
a₁,b₁ = T₁.((0.1,1.0))
using Random
Random.seed!(1234); 
rand_vals = rand(T₁,epsilon^2)
vals_epsilon = repeat(reshape(a₁ .+ (b₁-a₁)*rand_vals, (epsilon, epsilon)), inner=repeat_dims)

# using DelimitedFiles
# vals_epsilon = readdlm("./coeff_1.txt")

filename = project_dir*"/"*project_name*"/$(project_name)_params.csv"
write_problem_parameters(domain, nf, nc, p, l, ntimes, vals_epsilon, tf, Δt, filename)
