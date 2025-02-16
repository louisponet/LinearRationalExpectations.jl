using LinearAlgebra.LAPACK: geqrf!, ormqr!
using FastLapackInterface
using PolynomialMatrixEquations
const PME = PolynomialMatrixEquations
using PolynomialMatrixEquations: GSSolverWs, CRSolverWs
using SparseArrays
#using SolveEyePlusMinusAkronB: EyePlusAtKronBWs, generalized_sylvester_solver!

using LinearAlgebra.BLAS

"""
    Indices(n_exogenous::Int, forward::Vector{Int}, current::Vector{Int}, backward::Vector{Int}, static::Vector{Int})

Representation of the indices of the variables in matrices.
"""
struct Indices
    current        ::Vector{Int}
    forward        ::Vector{Int}
    purely_forward ::Vector{Int}
    backward       ::Vector{Int}
    both           ::Vector{Int}
    non_backward   ::Vector{Int} #union(purely_forward, static)
    
    static          ::Vector{Int}
    dynamic         ::Vector{Int}
    dynamic_current ::Vector{Int}
    
    current_in_dynamic ::Vector{Int} #ids of dynamic that are current
    forward_in_dynamic ::Vector{Int} #ids of dynamic that are forward
    backward_in_dynamic::Vector{Int} #ids of dynamic that are backward

    current_in_dynamic_jacobian ::Vector{Int}
    current_in_static_jacobian  ::Vector{Int}
    
    exogenous   ::Vector{Int}
    n_endogenous ::Int

    D_columns::NamedTuple{(:D, :jacobian), NTuple{2, Vector{Int}}}
    E_columns::NamedTuple{(:E, :jacobian), NTuple{2, Vector{Int}}}
    UD_columns::Vector{Int}
    UE_columns::Vector{Int}
    
end

function Indices(n_exogenous::Int, forward::Vector{Int}, current::Vector{Int}, backward::Vector{Int}, static::Vector{Int})
    n_forward  = length(forward)
    n_backward = length(backward)
    n_current  = length(current)
    
    n_endogenous = maximum(Iterators.flatten((forward, backward, current)))
    exogenous = n_backward + n_current + n_forward .+ (1:n_exogenous)
    
    both            = intersect(forward, backward)
    dynamic         = setdiff(collect(1:n_endogenous), static)
    current_dynamic = setdiff(current, static)
    purely_forward  = setdiff(forward, both)
    non_backward    = sort(union(purely_forward, static))
    
    forward_in_dynamic          = findall(in(forward), dynamic)
    backward_in_dynamic         = findall(in(backward), dynamic)
    current_in_dynamic          = findall(in(current_dynamic), dynamic)
    current_in_dynamic_jacobian = n_backward .+ findall(in(dynamic), current)
    current_in_static_jacobian  = n_backward .+ [findfirst(isequal(x), current) for x in static]

    # derivatives of current values of variables that are both
    # forward and backward are included in the D matrix
    k1 = findall(in(current), backward)
    k2a = findall(in(purely_forward), forward)
    k2b = findall(in(purely_forward), current)

    D_columns = (D        = [k1; n_backward .+ (1:n_forward)],
                 jacobian = [n_backward .+ findall(in(backward), current);
                             n_backward + n_current .+ (1:n_forward)])

    E_columns = (E        = [1:n_backward; n_backward .+ k2a],
                 jacobian = [1:n_backward; n_backward .+ k2b])
    
    UD_columns = findall(in(forward), backward)
    UE_columns = n_backward .+ findall(in(backward), forward)
    
    return Indices(
        current,
        forward,
        purely_forward,
        backward,
        both,
        non_backward,

        static,
        dynamic,
        current_dynamic,

        current_in_dynamic,
        forward_in_dynamic,
        backward_in_dynamic,
        current_in_dynamic_jacobian,
        current_in_static_jacobian,
        
        exogenous,
        n_endogenous,

        D_columns,
        E_columns,
        UD_columns,
        UE_columns,
    )
end

n_static(i::Indices)     = length(i.static)
n_forward(i::Indices)    = length(i.forward)
n_backward(i::Indices)   = length(i.backward)
n_both(i::Indices)       = length(i.both)
n_current(i::Indices)    = length(i.current)
n_dynamic(i::Indices)    = length(i.dynamic)
n_endogenous(i::Indices) = i.n_endogenous
n_exogenous(i::Indices) = length(i.exogenous)

"""
    LinearGSWs

Workspace used when solving systems with a dense representation of the jacobian. Uses Generalized Schur Decomposition when performing the linear solve.
Can be constructed with an [`Indices`](@ref).
"""
mutable struct LinearGSWs
    solver_ws::GSSolverWs
    ids::Indices
    d::Matrix{Float64}
    e::Matrix{Float64}
    jacobian_static::Matrix{Float64} 
    qr_ws::QRWs
   
    A_s::Matrix{Float64}
    C_s::Matrix{Float64}
    Gy_forward::Matrix{Float64}
    Gy_dynamic::Matrix{Float64}
    temp::Matrix{Float64}
    AGplusB_backward::Matrix{Float64}
    jacobian_forward::Matrix{Float64}
    jacobian_current::Matrix{Float64}
    b10::Matrix{Float64}
    b11::Matrix{Float64}
    AGplusB::Matrix{Float64}
    linsolve_static_ws::LUWs
    AGplusB_linsolve_ws::LUWs
end
function LinearGSWs(ids::Indices)
    n_back   = n_backward(ids)
    de_order = n_forward(ids) + n_back
    
    d = zeros(de_order, de_order)
    e = similar(d)
    
    solver_ws = GSSolverWs(d, n_back)
    
    n_stat = n_static(ids)
    n_forw = n_forward(ids)
    n_end  = n_endogenous(ids)
    n_curr = n_current(ids)
    
    jacobian_static = Matrix{Float64}(undef, n_endogenous(ids), n_static(ids))
     
    qr_ws = QRWs(jacobian_static)
    
    A_s = Matrix{Float64}(undef, n_stat, n_forw)
    C_s = Matrix{Float64}(undef, n_stat, n_back)
    
    Gy_forward = Matrix{Float64}(undef, n_forw, n_back)
    Gy_dynamic = Matrix{Float64}(undef, n_end - n_stat, n_back)
    
    temp = Matrix{Float64}(undef, n_stat, n_back)

    AGplusB_backward = Matrix{Float64}(undef, n_end, n_back)

    jacobian_forward = Matrix{Float64}(undef, n_end, n_forw)
    jacobian_current = Matrix{Float64}(undef, n_end, n_curr)

    b10 = Matrix{Float64}(undef, n_stat, n_stat)
    b11 = Matrix{Float64}(undef, n_stat, n_end - n_stat)
    
    linsolve_static_ws = LUWs(n_stat)
    
    AGplusB = Matrix{Float64}(undef, n_end, n_end)
    AGplusB_linsolve_ws = LUWs(n_end)
    
    return LinearGSWs(solver_ws, ids, d, e, jacobian_static, qr_ws,  A_s, C_s, Gy_forward, Gy_dynamic, 
        temp, AGplusB_backward, jacobian_forward, jacobian_current, b10, b11, AGplusB,
        linsolve_static_ws, AGplusB_linsolve_ws)
end

"""
    LinearCRWs

Workspace used when solving systems with a sparse representation of the jacobian. Uses Cyclic Reduction when performing the linear solve.
Can be constructed with an [`Indices`](@ref).
"""
mutable struct LinearCRWs
    solver_ws::CRSolverWs
    ids::Indices
    a::SparseMatrixCSC{Float64}
    b::Matrix{Float64}
    c::SparseMatrixCSC{Float64}
end

function LinearCRWs(ids::Indices)
    n = n_endogenous(ids)
    a = spzeros(n, n)
    b = Matrix{Float64}(undef, n, n)
    c = spzeros(n, n)
    solver_ws = CRSolverWs(a)
    return LinearCRWs(solver_ws, ids, a, b, c)
end

LinearRationalExpectationsWs(algo::String, ids::Indices) = 
    algo == "GS" ? LinearGSWs(ids) : LinearCRWs(ids)

LinearRationalExpectationsWs(algo::String, args...) = 
    LinearRationalExpectationsWs(algo, Indices(args...))

"""
    CROptions

Stores the options used during linear solving of sparse jacobians with [`LinearCRWs`](@ref).
"""
Base.@kwdef struct CROptions
    maxiter::Int = 100
    tol::Float64   = 1e-8
end

"""
    GSOptions

Stores the options used during linear solving of dense jacobians with [`LinearGSSolverWs`](@ref).
"""
Base.@kwdef struct GSOptions
    # Near unit roots are considered stable roots
    criterium::Float64 = 1.0 + 1e-6
end

"""
    LinearRationalExpectationsOptions

Holds both [`GeneralizedSchurOptions`](@ref) and [`CROptions`](@ref).
"""
Base.@kwdef struct LinearRationalExpectationsOptions
    cyclic_reduction::CROptions = CROptions()
    generalized_schur::GSOptions = GSOptions()
end

"""
    LinearRationalExpectationsResults

Stores the results of a linear solve.
see [Dynare working paper](https://www.dynare.org/wp-repo/dynarewp002.pdf) for more info.
"""
mutable struct LinearRationalExpectationsResults
    eigenvalues ::Vector{Complex{Float64}}
    g1          ::Matrix{Float64}  # full approximation
    gs1         ::Matrix{Float64} # state transition matrices: states x states
    hs1         ::Matrix{Float64} # states x shocks
    gns1        ::Matrix{Float64} # non states x states
    hns1        ::Matrix{Float64} # non states x shocsks
    
    # solution first order derivatives w.r. to state variables
    g1_1 ::SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
    # solution first order derivatives w.r. to current exogenous variables
    g1_2 ::SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
    
    endogenous_variance  ::Matrix{Float64}
    stationary_variables ::Vector{Bool}
    
    function LinearRationalExpectationsResults(n_endogenous::Int,
                                               exogenous_nbr::Int,
                                               backward_nbr::Int)
                                               
        state_nbr = backward_nbr + exogenous_nbr
        non_backward_nbr = n_endogenous - backward_nbr
        
        eigenvalues = Vector{Float64}(undef, 0)
        
        g1   = zeros(n_endogenous,(state_nbr + 1))
        gs1  = zeros(backward_nbr,backward_nbr)
        hs1  = zeros(backward_nbr,exogenous_nbr)
        
        gns1 = zeros(non_backward_nbr,backward_nbr)
        hns1 = zeros(non_backward_nbr,exogenous_nbr)
        
        g1_1 = view(g1, :, 1:backward_nbr)
        g1_2 = view(g1, :, backward_nbr .+ (1:exogenous_nbr))
        
        endogenous_variance  = zeros(n_endogenous, n_endogenous)
        stationary_variables = Vector{Bool}(undef, n_endogenous)
        
        new(eigenvalues, g1, gs1, hs1, gns1, hns1, g1_1, g1_2, endogenous_variance, stationary_variables)
    end
end

function copy_jacobian!(ws::LinearGSWs,
                        jacobian::AbstractMatrix{Float64})
    fill!(ws.d, 0.0)
    fill!(ws.e, 0.0)

    ids = ws.ids
    
    n_dyn  = n_dynamic(ids)
    dyn_r  = 1:n_dyn
    r      = n_static(ids) .+ dyn_r
    both_r = 1:n_both(ids)
    
    @views @inbounds begin
        
        ws.d[dyn_r, ids.D_columns.D] .=     jacobian[r, ids.D_columns.jacobian]
        ws.e[dyn_r, ids.E_columns.E] .=  .- jacobian[r, ids.E_columns.jacobian]
        
        for i = both_r 
            k = n_dyn + i
            
            ws.d[k, ids.UD_columns[i]] = 1.0
            ws.e[k, ids.UE_columns[i]] = 1.0
        end
        
    end
    return ws.d, ws.e
end

"""
    remove_static!(jacobian::Matrix{Float64}, ws::LinearGSSolverWs)

Removes a subset of variables (columns) and rows by QR decomposition.

`jacobian`:
    - on entry: jacobian matrix of the original model
    - on exit:  transformed jacobian -- the rows corresponding to the dynamic part are at the bottom
`ws`: on exit contains the triangular part conrresponding to static variables
"""
function remove_static!(jacobian::Matrix{Float64},
                        ws::LinearGSWs)
    ws.jacobian_static .= view(jacobian, :, ws.ids.current_in_static_jacobian)
    geqrf!(ws.qr_ws, ws.jacobian_static)
    ormqr!(ws.qr_ws, 'L', 'T', ws.jacobian_static, jacobian)
end

"""
    add_static!(results, jacobian::Matrix{Float64}, ws::LinearGSSolverWs)
    
Computes the solution for the static variables:
```math
G_{y,static} = -B_{s,s}^{-1}(A_s G{y,fwrd} Gs + B_{s,d} G_{y,dynamic} + C_s)
```
""" 
function add_static!(results::LinearRationalExpectationsResults,
                     jacobian::Matrix{Float64},
                     ws::LinearGSWs)
    ids = ws.ids
    @views @inbounds begin
        # static rows are at the top of the QR transformed Jacobian matrix
        stat_r = 1:n_static(ids)
        back_r = 1:n_backward(ids)
        # B_s,s
        # fill!(ws.b10, 0.0)
        ws.b10 .= jacobian[stat_r, ids.current_in_static_jacobian]
        # B_s,d
        ws.b11[:, ids.current_in_dynamic] .= jacobian[stat_r, ids.current_in_dynamic_jacobian]
        # A_s
        ws.A_s .= jacobian[stat_r, n_backward(ids) + n_current(ids) .+ (1:n_forward(ids))]
        # C_s
        ws.C_s .= jacobian[stat_r, back_r]
        # Gy.fwrd
        ws.Gy_forward .= results.g1_1[ids.forward, back_r]
        # Gy.dynamic
        ws.Gy_dynamic .= results.g1_1[ids.dynamic, back_r]
        # ws.C_s = B_s,d*Gy.dynamic + C_s
        
        mul!(ws.C_s, ws.b11, ws.Gy_dynamic, 1.0, 1.0)
        # ws.temp = A_s*Gy.fwrd*gs1
        mul!(ws.temp, ws.A_s, ws.Gy_forward)
        mul!(ws.C_s, ws.temp, results.gs1, -1.0, -1.0)
        # ws.Gy_forward = B_s,s\ws.C_s
        
        lu_t = LU(factorize!(ws.linsolve_static_ws, ws.b10)...)
        ldiv!(lu_t, ws.C_s)
        results.g1[ids.static, back_r] .= ws.C_s
    end
    return results.g1, jacobian
end

function make_AGplusB!(AGplusB::AbstractMatrix{Float64},
                          A::AbstractMatrix{Float64},
                          G::AbstractMatrix{Float64},
                          B::AbstractMatrix{Float64},
                          ws::LinearGSWs)
                          
    fill!(AGplusB, 0.0)
    ids = ws.ids
    
    @views @inbounds begin
        AGplusB[:, ids.current] .= B
        ws.Gy_forward .= G[ids.forward, :]
        mul!(ws.AGplusB_backward, A, ws.Gy_forward)
        AGplusB[:, ids.backward] .+= ws.AGplusB_backward
        return AGplusB
    end
    
end

function solve_for_derivatives_with_respect_to_shocks!(results::LinearRationalExpectationsResults,
                                                       jacobian::AbstractMatrix{Float64},
                                                       ws::LinearGSWs)
    #=
    if model.lagged_exogenous_nbr > 0
        f6 = view(jacobian,:,model.i_lagged_exogenous)
        for i = 1:model.current_exogenous_nbr
            for j = 1:model.endo_nbr
                results.g1_3[i,j] = -f6[i,j]
            end
        end
        linsolve_core_no_lu!(results.f1g1plusf2, results.g1_3, ws)
    end
    =#
    if n_exogenous(ws.ids) > 0
        results.g1_2 .= .-view(jacobian, :, ws.ids.exogenous)
        
        lu_t = LU(factorize!(ws.AGplusB_linsolve_ws, ws.AGplusB)...)
        
        ldiv!(lu_t, results.g1_2)
        
    end
end

"""
    solve!(results::LinearRationalExpectationsResults, jacobian::Matrix, options, ws::LinearGSSolverWs)
    solve!(results::LinearRationalExpectationsResults, jacobian::SparseMatrixCSC, options, ws::LinearCRWs)

Solve the linear rational expectation system.
See [Dynare working paper](https://www.dynare.org/wp-repo/dynarewp002.pdf) for more info.
"""
function solve!(results::LinearRationalExpectationsResults, jacobian::Matrix, options, ws::LinearGSWs)
    
    ids       = ws.ids
    back      = ids.backward
    n_back    = n_backward(ids)
    n_cur     = n_current(ids)
    n_for     = n_forward(ids)
    
    forward_r = 1:n_for
    current_r = 1:n_cur
    back_r    = 1:n_back
    pur_for   = ids.purely_forward

    
    remove_static!(jacobian, ws)
    copy_jacobian!(ws, jacobian)
    
    try
        PME.solve!(ws.solver_ws, ws.d, ws.e; tolerance= 1-options.generalized_schur.criterium)
    finally
        resize!(results.eigenvalues, length(ws.solver_ws.schurws.eigen_values))
        copy!(results.eigenvalues, ws.solver_ws.schurws.eigen_values)
    end
    
    results.gs1                 .= ws.solver_ws.g1
    results.g1[back, back_r]    .= ws.solver_ws.g1[back_r, back_r]
    results.g1[pur_for, back_r] .= ws.solver_ws.g2[ids.E_columns.E[n_back .+ (1:(n_for - n_both(ids)))] .- n_back, :]

    if n_static(ids) > 0
        results.g1, jacobian = add_static!(results, jacobian, ws)
    end
    
    ws.jacobian_forward[:, forward_r] .= jacobian[:, n_back + n_cur .+ forward_r]
    ws.jacobian_current[:, current_r] .= jacobian[:, n_back .+ current_r]
   
    ws.AGplusB = make_AGplusB!(ws.AGplusB, ws.jacobian_forward, results.g1_1, ws.jacobian_current, ws)
    
    solve_for_derivatives_with_respect_to_shocks!(results, jacobian, ws)
    
    fill_results!(results, ids)
    return results
end

function solve!(results::LinearRationalExpectationsResults, jacobian::SparseMatrixCSC, options, ws::LinearCRWs)
    ids     = ws.ids
    back_r  = 1:n_backward(ids)
    dyn     = ids.dynamic
    back_d  = ids.backward_in_dynamic
    
    n       = ids.n_endogenous
    
    @inbounds @views begin
        fill!(ws.a, 0.0)
        fill!(ws.b, 0.0)
        fill!(ws.c, 0.0)
        
        ws.a .= jacobian[:, 2n + 1:3n]
        ws.b .= jacobian[:, n+1:2n]
        ws.c .= jacobian[:, 1:n]
        
        PME.solve!(ws.solver_ws,  ws.c, ws.b, ws.a; tolerance=options.cyclic_reduction.tol, max_iterations=options.cyclic_reduction.maxiter)
        
        results.gs1[:, back_r] .= ws.solver_ws.x[ids.backward, ids.backward]
        results.g1[:, back_r]  .= ws.solver_ws.x[:, ids.backward]
        results.g1_2           .= .-(ws.a * ws.solver_ws.x + ws.b) \ jacobian[:,3n+1:end]
    end
    
    fill_results!(results, ids)
    return results
end

function fill_results!(results::LinearRationalExpectationsResults, ids::Indices)
    @views @inbounds begin
        results.hs1  .= results.g1_2[ids.backward, :]
        results.gns1 .= results.g1_1[ids.non_backward, :]
        results.hns1 .= results.g1_2[ids.non_backward, :]
    end
end
