######### ######### ######### ######### ######### ######### ######### ######### #########
# Read the basis functions from the files and then construct the multiscale system
######### ######### ######### ######### ######### ######### ######### ######### #########

include("./fileIO.jl");
include("./time-dependent.jl")

# Load all the params
project_dir, project_name, ntimes_1 = ARGS;
param_filename = project_dir*"/"*project_name*"/$(project_name)_params.csv";
domain, nf, nc, p, l, ntimes, vals_epsilon, tf, Δt, T₁ = read_problem_parameters(param_filename);

ntimes = parse(Int64, ntimes_1)

@assert ntimes==0 "Run this script only for j=0"

##### ##### ##### ##### ##### ##### #####
# Temporal discretization scheme
##### ##### ##### ##### ##### ##### #####
ntime = ceil(Int, tf/Δt)
BDF = 4

# Define the RHS and the initial condition
f(x,t) = T₁(10*2π^2*sin(π*x[1])*sin(π*x[2])*(sin(t))^5)
u₀(x) = T₁(0.0)

# Background fine scale discretization
FineScale = FineTriangulation(domain, nf);
reffe = ReferenceFE(lagrangian, T₁, 1);
V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1; vector_type=Vector{T₁});

A = CellField(vec(vals_epsilon), FineScale.trian)

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);

# Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);

# Assemble the fine scale matrices
K = assemble_stima(V₀, A, 4; T=T₁);
M = assemble_massma(V₀, x->1.0, 4; T=T₁);
L = assemble_rect_matrix(Ωₘₛ, p);
Λ = assemble_lm_l2_matrix(Ωₘₛ, p);

function load_basis!(γₘₛ)
    @showprogress desc="Loading MS Bases..." for i=2:nc*nc
        filename = project_dir*"/"*project_name*"/$(project_name)_ms_basis_$(nc)$(p)$(l)_"*string(i)*".csv"
        γₘₛ += read_basis_functions(filename, T₁, size(L))
    end
    γₘₛ
end


fname_1 = project_dir*"/"*project_name*"/$(project_name)_ms_basis_$(nc)$(p)$(l)_"*string(1)*".csv"
γₘₛ = read_basis_functions(fname_1, T₁, size(L))
γₘₛ = load_basis!(γₘₛ)

###### ###### ###### ###### ###### ###### ###### ###### ###### ######
# Compute the matrix system using the basis functions
###### ###### ###### ###### ###### ###### ###### ###### ###### ######
Kₘₛ = γₘₛ'*K*γₘₛ
Mₘₛ = γₘₛ'*M*γₘₛ

M = Mₘₛ
K = Kₘₛ

println("Solving multiscale problem...")
function fₙ(cache, tₙ::Float64)
    Vₕ, B = cache
    L = assemble_loadvec(Vₕ, y->f(y,tₙ), 8; T=T₁)
    B'*L
end

let
    U₀ = setup_initial_condition(u₀, γₘₛ, V₀; T=T₁)
    global U = zero(U₀)
    t = 0.0
    # Starting BDF steps (1...k-1)
    fcache = (V₀, γₘₛ)
    @showprogress desc="Solving MS system 1 to $(BDF-1) ..." for i=1:BDF-1
        dlcache = get_dl_cache(i)
        cache = dlcache, fcache
        U₁ = BDFk!(cache, t, U₀, Δt, K, M, fₙ, i)
        U₀ = hcat(U₁, U₀)
        t += Δt
    end
    # Remaining BDF steps
    dlcache = get_dl_cache(BDF)
    cache = dlcache, fcache
    @showprogress desc="Solving MS system $(BDF) to $ntime ..." for i=BDF:ntime
        U₁ = BDFk!(cache, t+Δt, U₀, Δt, K, M, fₙ, BDF)
        U₀[:,2:BDF] = U₀[:,1:BDF-1]
        U₀[:,1] = U₁
        t += Δt
    end
    U = U₀[:,1] # Final time solution
end

using DataFrames, CSV
CSV.write(project_dir*"/"*project_name*"/$(project_name)_ms_solution_raw.csv", DataFrame((a=U)))

Uₘₛ = γₘₛ*U[1:(p+1)^2*num_cells(CoarseScale.trian)]
CSV.write(project_dir*"/"*project_name*"/$(project_name)_ms_solution.csv", DataFrame((a=Uₘₛ)))

Uref = CSV.read(project_dir*"/"*project_name*"/$(project_name)_ref_solution_$nf.csv", DataFrame, types=[T₁]).a

uₐ = FEFunction(V₀, Uₘₛ)
uₑ = FEFunction(V₀, Uref)
err = uₑ - uₐ

Ωf = FineScale.trian
dΩf = Measure(Ωf, 4)
L²Error = sqrt(sum( ∫((err)*(err))dΩf ))
H¹Error = sqrt(sum(∫(A*(∇(err))⊙(∇(err)))dΩf))
# println("L²Error = $L²Error, \t H¹Error = $H¹Error");
println("$nf \t $nc \t $p \t $l \t $ntimes \t $L²Error \t $H¹Error")

error_data = Dict(:nf=>nf, 
                  :nc=>nc,
                  :p=>p,
                  :l=>l,
                  :ntimes=>ntimes,
                  :l2error=>L²Error,
                  :h1error=>H¹Error);
CSV.write(project_dir*"/"*project_name*"/$(project_name)_error_data.csv", DataFrame(error_data))
