import warnings

from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from skglm import Lasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "skglm"
    stopping_strategy = "iteration"

    install_cmd = "conda"
    requirements = ["pip:git+https://github.com/mathurinm/skglm@main"]
    references = [
        "Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel"
        "and M. Massias"
        '"Beyond L1: Faster and Better Sparse Models with skglm", '
        "https://arxiv.org/abs/2204.07826"
    ]

    def set_objective(self, X, y, lambdas, fit_intercept, n_lambda):
        self.X = X
        self.y = y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept
        self.n_lambda = n_lambda
        self.lasso = Lasso()

        # Trigger numba JIT compilation
        self.run(1)

    def skip(self, X, y, lambdas, fit_intercept, n_lambda):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def run(self, n_iter):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        _, self.coefs, _, _ = self.lasso.path(
            self.X,
            self.y,
            alphas=self.lambdas / len(self.y),
            tol=1e-12,
            max_iter=n_iter,
            max_epochs=100_000,
        )

    @staticmethod
    def get_next(previous):
        return previous + 1

    def get_result(self):
        return self.coefs
