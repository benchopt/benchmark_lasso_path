import warnings

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from celer import Lasso, celer_path
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "Celer"

    install_cmd = "conda"
    requirements = ["pip:celer"]
    references = [
        "M. Massias, A. Gramfort and J. Salmon, ICML, "
        '"Celer: a Fast Solver for the Lasso with Dual Extrapolation", '
        "vol. 80, pp. 3321-3330 (2018)"
    ]

    def set_objective(self, X, y, lambdas, fit_intercept, n_lambda, lambda_min_ratio):
        self.X, self.y = X, y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept
        self.n_lambda = n_lambda

    def skip(self, X, y, lambdas, fit_intercept, n_lambda, lambda_min_ratio):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def run(self, n_iter):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        _, self.coefs, _ = celer_path(
            self.X,
            self.y,
            "lasso",
            alphas=self.lambdas / len(self.y),
            prune=1,
            tol=1e-12,
            max_iter=n_iter,
            max_epochs=100_000,
        )

    @staticmethod
    def get_next(previous):
        "Linear growth for n_iter."
        return previous + 1

    def get_result(self):
        return self.coefs
