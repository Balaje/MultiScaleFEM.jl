##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Functions to perform the read/write operations
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

using Gridap
using HigherOrderMS_2d
using SparseArrays
using ProgressMeter

using DataFrames, CSV

"""
Function to write the problem parameters into the project folder
"""
function write_problem_parameters(domain::NTuple{4,T3}, nf::T, nc::T,
                                  p::T, l::T, ntimes::T,
                                  coeffs::AbstractVecOrMat{T3},
                                  tf::T1, Δt::T1,
                                  filename::T2) where
    {T<:Integer, T1<:Real, T2<:String, T3<:Real}
    df = DataFrame([:domain => domain, :nf => nf, :nc => nc, :p => p, :l => l,
                    :j => ntimes, :tf => tf, :dt => Δt,
                    :type => T3])
    CSV.write(filename, df)
    df = DataFrame(coeffs, :auto)
    CSV.write(filename[1:end-4]*"_coeffs.csv", df)
end

"""
Function to read the problem parameters from the project folder
         to compute the basis functions
"""
function read_problem_parameters(filename::T1) where {T1<:String}
    B = CSV.read(filename, DataFrame)
    el_type = eval(Meta.parse(B.type[1]))
    domain = el_type.(eval(Meta.parse(B.domain[1])))
    nc = B.nc[1]
    nf = B.nf[1]
    p = B.p[1]
    l = B.l[1]
    ntimes = B.j[1]
    tf = B.tf[1]
    dt = B.dt[1]
    coeffs = CSV.read(filename[1:end-4]*"_coeffs.csv", DataFrame)
    (domain, nf, nc, p, l, ntimes, vec(Matrix(coeffs)), tf, dt, el_type)
end

"""
Function to write the basis functions into the project folder
"""
function write_basis_functions(V, FILENAME)
    I, J, VALS = findnz(V)
    df = DataFrame((a=I, b=J, c=VALS))
    CSV.write(FILENAME, df)
end

"""
Function to read the basis functions from the project folder to compute the multiscale system
"""
function read_basis_functions(FILENAME, ::Type{T}, mat_size) where T<:Real
    B = CSV.read(FILENAME, DataFrame, types=[Int64, Int64, String]);
    sparse(B.a, B.b, parse.(T,B.c), mat_size...)
end
