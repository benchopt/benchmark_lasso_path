from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import lars_path


class Solver(BaseSolver):
    name = "LARS"
    support_sparse = False

    install_cmd = "conda"
    requirements = ["scikit-learn"]
    references = [
        "B. Efron, T. Hastie, I. Johnstone, R. Tibshirani"
        '"Least Angle Regression", Annals of Statistics, '
        " vol. 32 (2), pp. 407-499 (2004)"
    ]

    def set_objective(self, X, y, lambdas, fit_intercept):
        self.X = X
        self.y = y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

    def skip(self, X, y, lambdas, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def run(self, n_iter):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # XXX ISSUE #1: n_iter doesn't make sense for LARS.
        # The parameter max_iter refers to # of nodes/breakpoints in the path
        # If max_iter<np.inf, LARS might not compute the path until alphas[-1]
        # Another option would be to set max_iter=np.inf and use a stopping
        # strategy based Cholesky precision by setting the argument 'eps'
        # However, it does not seem to lead to increasing running times
        self.alphas, _, self.coefs = lars_path(
            self.X,
            self.y,
            alpha_min=self.lambdas[-1] / len(self.y),
            max_iter=n_iter,
            method="lasso",
            return_path=True,
        )

    def get_result(self):
        # XXX TO IMPROVE: find coeficients at regularization in self.lambdas
        # coefs returned by lasso_path are sampled at the nodes of the path
        # coefficients between two nodes can be found by linear interpolation

        # Getting values at lambdas grid via interpolation done coordinate
        # by coordinate with np.interp.
        # There should be a more efficient way of doing this.
        # Also np.interp requires increasing inputs,
        # so we need to reverse all arrays
        coefs = np.array(
            [
                np.flip(
                    np.interp(
                        self.lambdas[::-1] / len(self.y),
                        self.alphas[::-1],
                        self.coefs[i, ::-1],
                    )
                )
                for i in range(self.coefs.shape[0])
            ]
        )

        return coefs
