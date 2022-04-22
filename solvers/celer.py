import warnings

from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from celer import celer_path
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "Celer"
    stopping_strategy = "tolerance"

    install_cmd = "conda"
    requirements = ["pip:celer"]
    references = [
        "M. Massias, A. Gramfort and J. Salmon, ICML, "
        '"Celer: a Fast Solver for the Lasso with Dual Extrapolation", '
        "vol. 80, pp. 3321-3330 (2018)"
    ]

    def set_objective(self, X, y, lambdas, fit_intercept):
        self.X, self.y = X, y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

    def skip(self, X, y, lambdas, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def run(self, tol):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        _, self.coefs, _ = celer_path(
            self.X,
            self.y,
            pb="lasso",
            alphas=self.lambdas / len(self.y),
            prune=1,
            tol=tol,
            max_iter=1_000,
            max_epochs=100_000,
        )

    def get_result(self):
        return self.coefs
