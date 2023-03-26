using Random
using LinearRationalExpectations: n_backward, n_forward, n_both, n_static, n_dynamic, n_current, n_exogenous
using LinearRationalExpectations.SparseArrays


using DelimitedFiles

const LRE = LinearRationalExpectations

jacobian = readdlm(joinpath(@__DIR__, "models/test/jacobian.txt"))
exogenous_nbr = 101
forward_indices  = readdlm(joinpath(@__DIR__, "models/test/i_fwrd_b.txt"),  Int)[:, 1]
current_indices  = readdlm(joinpath(@__DIR__, "models/test/i_current.txt"), Int)[:, 1]
backward_indices = readdlm(joinpath(@__DIR__, "models/test/i_bkwrd_b.txt"), Int)[:, 1]
static_indices   = readdlm(joinpath(@__DIR__, "models/test/i_static.txt"),  Int)[:, 1]

lli = readdlm(joinpath(@__DIR__, "models/test/lli.txt"), Int)

ids = LRE.Indices(exogenous_nbr, forward_indices, current_indices, backward_indices, static_indices)

J = hcat(jacobian[:, findall(lli[1, :] .> 0)],
                jacobian[:, LRE.n_endogenous(ids) .+ findall(lli[2, :] .> 0)],
                jacobian[:, 2*LRE.n_endogenous(ids) .+ findall(lli[3, :] .> 0)],
                jacobian[:, 3*LRE.n_endogenous(ids) .+ collect(1:LRE.n_exogenous(ids))])

n_back = n_backward(ids)
back_r = 1:n_back

results = LRE.LinearRationalExpectationsResults(LRE.n_endogenous(ids), LRE.n_exogenous(ids), LRE.n_backward(ids))

results_gs = nothing
@testset "GS" begin
    ws = LRE.LinearGSWs(ids)
    
    gs_jacobian = Matrix(J)

    LRE.remove_static!(gs_jacobian, ws) 
    LRE.copy_jacobian!(ws, gs_jacobian)
    d_orig = copy(ws.d)
    e_orig = copy(ws.e)

    @testset "solving" begin
        options = LinearRationalExpectationsOptions()
        
        LRE.solve!(results, Matrix(J), options, ws)
        
        @test d_orig * vcat(I(n_back), ws.solver_ws.g2[:, back_r])*ws.solver_ws.g1 ≈ e_orig * vcat(I(n_back), ws.solver_ws.g2[:, back_r])
        global results_gs =deepcopy(results)
    end

end

@testset "CR" begin
    ws = LRE.LinearCRWs(ids)
    
    results = LRE.LinearRationalExpectationsResults(LRE.n_endogenous(ids), LRE.n_exogenous(ids), LRE.n_backward(ids))
                                                
    options = LinearRationalExpectationsOptions()
    @testset "solving" begin

        LRE.solve!(results, sparse(jacobian), options, ws)

        @test results_gs.g1 ≈ results.g1
        @test results_gs.gs1 ≈ results.gs1
        @test results_gs.hs1 ≈ results.hs1
        @test results_gs.gns1 ≈ results.gns1
        @test results_gs.hns1 ≈ results.hns1
    end    
   
end
