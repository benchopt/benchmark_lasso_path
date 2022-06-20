using Lasso
using PyCall
using SparseArrays

function scipyCSC_to_julia(A)
    m, n = A.shape
    colPtr = Int[i + 1 for i in PyArray(A."indptr")]
    rowVal = Int[i + 1 for i in PyArray(A."indices")]
    nzVal = Vector{Float64}(PyArray(A."data"))
    B = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)

    return B
end

function solve_lasso(
    X,
    y::Vector{Float64},
    lambda::Vector{Float64},
    fit_intercept::Bool,
    tol::Float64,
    cd_maxiter::Int,
    get_null_solution::Bool,
)
    p = size(X, 2)

    w_nrow = if fit_intercept p + 1 else p end

    w = zeros(Float64, w_nrow, length(lambda))

    if !get_null_solution
        lasso_fit = fit(
            LassoPath,
            X,
            y;
            λ=lambda,
            stopearly=false,
            standardize=false,
            intercept=fit_intercept,
            maxncoef=max(size(X, 1), size(X, 2)) * 100,
            cd_tol=tol,
            cd_maxiter=cd_maxiter
        )
        w = coef(lasso_fit)
    end

    return w
end
