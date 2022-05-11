from pathlib import Path

import numpy as np
from benchopt import safe_import_context
from benchopt.helpers.julia import (JuliaSolver, assert_julia_installed,
                                    get_jl_interpreter)
from benchopt.runner import INFINITY
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    assert_julia_installed()
    from scipy import sparse


# File containing the function to be called from julia
JULIA_SOLVER_FILE = str(Path(__file__).with_suffix(".jl"))


class Solver(JuliaSolver):
    name = "lasso_jl"
    stopping_criterion = SufficientProgressCriterion(
        patience=7, eps=1e-15, strategy="tolerance"
    )
    julia_requirements = [
        "Distributions",
        "GLM",
        "Lasso",
        "LinearAlgebra",
        "PyCall",
        "SparseArrays",
    ]
    references = [
        'J. Friedman, T. J. Hastie and R. Tibshirani, "Regularization paths '
        'for generalized linear models via coordinate descent", '
        "J. Stat. Softw., vol. 33, no. 1, pp. 1-22, NIH Public Access (2010)"
    ]
    support_sparse = True

    def set_objective(self, X, y, lambdas, fit_intercept):
        self.n, self.p = X.shape
        self.X = X
        self.y = y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

        jl = get_jl_interpreter()
        jl.include(str(JULIA_SOLVER_FILE))
        self.solve_lasso = jl.solve_lasso

        if sparse.issparse(X):
            scipyCSC_to_julia = jl.pyfunctionret(
                jl.scipyCSC_to_julia, jl.Any, jl.PyObject
            )
            self.X = scipyCSC_to_julia(X)

        # Trigger Julia JIT compilation
        self.run(INFINITY)

    def run(self, tol):
        if tol == INFINITY:
            self.coefs = np.zeros((self.p, len(self.lambdas)))
            if self.fit_intercept:
                intercepts = np.empty(len(self.lambdas))
                intercepts.fill(np.mean(np.array(self.y)))
                self.coefs = np.vstack((intercepts, self.coefs))
        else:
            self.coefs = self.solve_lasso(
                self.X,
                self.y,
                self.lambdas / len(self.y),
                self.fit_intercept,
                tol**1.8,
            )

    def get_result(self):
        coefs = np.array(self.coefs)

        if self.fit_intercept:
            coefs = np.vstack((coefs[1:, :], coefs[0, :]))

        return coefs
