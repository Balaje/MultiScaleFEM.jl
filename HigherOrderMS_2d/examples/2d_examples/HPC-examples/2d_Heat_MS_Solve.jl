######### ######### ######### ######### ######### ######### ######### ######### #########
# Read the basis functions from the files and then construct the multiscale system
######### ######### ######### ######### ######### ######### ######### ######### #########

include("./fileIO.jl");
include("../time-dependent.jl")

# Load all the params
project_dir, project_name, ntimes_1 = ARGS;
param_filename = project_dir*"/"*project_name*"/$(project_name)_params.csv";
domain, nf, nc, p, l, ntimes, vals_epsilon, tf, Δt, T₁ = read_problem_parameters(param_filename);

ntimes = parse(Int64, ntimes_1)

##### ##### ##### ##### ##### ##### #####
# Temporal discretization scheme
##### ##### ##### ##### ##### ##### #####
ntime = ceil(Int, tf/Δt)
BDF = 4

# Define the RHS and the initial condition
f(x,t) = T₁(10*2π^2*sin(π*x[1])*sin(π*x[2])*(sin(t))^4)
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
function load_additional_corrections!(Wₘₛ)
    for j=1:ntimes
        @showprogress desc="Loading Additional Corrections $j..." for i=2:nc*nc
            filename = project_dir*"/"*project_name*"/$(project_name)_ms_basis_$(nc)$(p)$(l)_correction_level_$(j)_"*string(i)*".csv"
            Wₘₛ[j] += read_basis_functions(filename, T₁, size(L))
        end
    end
    Wₘₛ
end


fname_1 = project_dir*"/"*project_name*"/$(project_name)_ms_basis_$(nc)$(p)$(l)_"*string(1)*".csv"
γₘₛ = read_basis_functions(fname_1, T₁, size(L))
γₘₛ = load_basis!(γₘₛ)

fname_2(j) = project_dir*"/"*project_name*"/$(project_name)_ms_basis_$(nc)$(p)$(l)_correction_level_$(j)_"*string(1)*".csv"
Wₘₛ = [read_basis_functions(fname_2(j), T₁, size(L)) for j=1:ntimes]
Wₘₛ = load_additional_corrections!(Wₘₛ);
Wₘₛ = hcat(Wₘₛ...);

#=
###### ###### ###### ###### ###### ###### ###### ###### ###### ######
# Compute the matrix system using the basis functions
###### ###### ###### ###### ###### ###### ###### ###### ###### ######
Kₘₛ = γₘₛ'*K*γₘₛ
Mₘₛ = γₘₛ'*M*γₘₛ
Pₘₛ = γₘₛ'*K*Wₘₛ
Lₘₛ = γₘₛ'*M*Wₘₛ
Kₘₛ′ = Wₘₛ'*K*Wₘₛ
Mₘₛ′ = Wₘₛ'*M*Wₘₛ

sM = [Mₘₛ Lₘₛ; Lₘₛ' Mₘₛ′]
sK = [Kₘₛ Pₘₛ; Pₘₛ' Kₘₛ′]

write_basis_functions(sM, project_dir*"/"*project_name*"/$(project_name)_mass_matrix_correction_level_$(ntimes).csv")
write_basis_functions(sK, project_dir*"/"*project_name*"/$(project_name)_stiffness_matrix_correction_level_$(ntimes).csv")
=#

ms_problem_size = ( nc^2*(p+1)^2*(ntimes+1), nc^2*(p+1)^2*(ntimes+1) )
M = read_basis_functions(project_dir*"/"*project_name*"/$(project_name)_mass_matrix_correction_level_$(ntimes).csv", T₁, ms_problem_size)
K = read_basis_functions(project_dir*"/"*project_name*"/$(project_name)_stiffness_matrix_correction_level_$(ntimes).csv", T₁, ms_problem_size)

### ### ### ### ### ### ### ### ### ### ### ###
#  Construct the schur complement system
### ### ### ### ### ### ### ### ### ### ### ###
sM = SchurComplementMatrix( M, nc^2*(p+1)^2*ntimes, nc^2*(p+1)^2 )
sK = SchurComplementMatrix( K, nc^2*(p+1)^2*ntimes, nc^2*(p+1)^2 )

println("Solving multiscale problem...")
function fₙ(cache, tₙ::Float64)
    Vₕ, B, B₂ = cache
    L = assemble_loadvec(Vₕ, y->f(y,tₙ), 4; T=T₁)
    [B'*L; B₂'*L]
end

let
    U₀ = [setup_initial_condition(u₀, γₘₛ, V₀; T=T₁); zeros(T₁, ntimes*(p+1)^2*num_cells(CoarseScale.trian))]
    global U = zero(U₀)
    t = 0.0
    # Starting BDF steps (1...k-1)
    fcache = (V₀, γₘₛ, Wₘₛ)
    @showprogress desc="Time stepping 1 to $(BDF-1) ..." for i=1:BDF-1
        dlcache = get_dl_cache(i)
        cache = dlcache, fcache
        U₁ = BDFk!(cache, t, U₀, Δt, sK, sM, fₙ, i)
        U₀ = hcat(U₁, U₀)
        t += Δt
    end
    # Remaining BDF steps
    dlcache = get_dl_cache(BDF)
    cache = dlcache, fcache
    @showprogress desc="Time stepping $(BDF) to $ntime ..." for i=BDF:ntime
        U₁ = BDFk!(cache, t+Δt, U₀, Δt, sK, sM, fₙ, BDF)
        U₀[:,2:BDF] = U₀[:,1:BDF-1]
        U₀[:,1] = U₁
        t += Δt
    end
    U = U₀[:,1] # Final time solution
end

using DataFrames, CSV
CSV.write(project_dir*"/"*project_name*"/$(project_name)_ms_solution_raw.csv", DataFrame((a=U)))

Uₘₛ = Wₘₛ*U[(p+1)^2*num_cells(CoarseScale.trian)+1:end] + γₘₛ*U[1:(p+1)^2*num_cells(CoarseScale.trian)]
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
