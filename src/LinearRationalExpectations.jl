module LinearRationalExpectations

include("linear_rational_expectations.jl")
export LinearRationalExpectationsWs,
    LinearRationalExpectationsResults, solve!,
    LinearRationalExpectationsOptions, CROptions,
    GSOptions

include("extended_lyapunov.jl")
export LyapdWs, extended_lyapd!, extended_lyapd_core!, is_stationary

include("variance.jl")
export VarianceWs, compute_variance!, correlation, variance_decomposition,
    variance_decomposition!, autocovariance!, autocorrelation!

end    
