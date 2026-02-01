import warnings

from benchopt import BaseSolver
import numpy as np
from scipy import sparse
from skglm.datafits import Quadratic
from skglm.penalties import L1
from skglm.solvers import AndersonCD
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._base import _preprocess_data


class Solver(BaseSolver):
    name = "skglm"
    stopping_strategy = "iteration"
    support_sparse = False

    install_cmd = "conda"
    requirements = ["skglm>=0.5"]
    references = [
        "Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel"
        "and M. Massias"
        '"Beyond L1: Faster and Better Sparse Models with skglm", '
        "https://arxiv.org/abs/2204.07826"
    ]

    def skip(self, X, y, lambdas, fit_intercept):
        # skglm does not support yet passing X_offset along with non-centered X
        #  as done in sklearn for sparse data
        if fit_intercept and sparse.issparse(X):
            return (
                True,
                f"{self.name} doesn't handle fit_intercept with sparse data",
            )

        return False, None

    def set_objective(self, X, y, lambdas, fit_intercept):
        # sklearn way of handling intercept: center X and y for dense
        if fit_intercept:
            X, y, X_offset, y_offset, _, _ = _preprocess_data(
                X=X, y=y, fit_intercept=fit_intercept, copy=True
            )
            self.X_offset = X_offset
            self.y_offset = y_offset

        self.X = X
        self.y = y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

        self.datafit = Quadratic()
        self.penalty = L1(1.)
        self.solver = AndersonCD(
            fit_intercept=False, tol=1e-12, max_epochs=100_000)

    def warm_up(self):
        # Trigger numba JIT compilation
        self.run(1)

    def run(self, n_iter):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        self.solver.max_iter = n_iter
        _, self.coefs, _ = self.solver.path(
            self.X,
            self.y,
            self.datafit,
            self.penalty,
            alphas=self.lambdas / len(self.y),
        )

        if self.fit_intercept:
            intercept = self.y_offset - self.X_offset @ self.coefs
            self.coefs = np.vstack((self.coefs, intercept))

    @staticmethod
    def get_next(previous):
        return previous + 1

    def get_result(self):
        return dict(coefs=self.coefs)
