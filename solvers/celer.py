import warnings

from benchopt import BaseSolver
import numpy as np
from celer import celer_path
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._base import _preprocess_data


class Solver(BaseSolver):
    name = "Celer"
    stopping_strategy = "iteration"

    install_cmd = "conda"
    requirements = ["celer"]
    references = [
        "M. Massias, A. Gramfort and J. Salmon, ICML, "
        '"Celer: a Fast Solver for the Lasso with Dual Extrapolation", '
        "vol. 80, pp. 3321-3330 (2018)"
    ]

    def set_objective(self, X, y, lambdas, fit_intercept):
        # celer/sklearn way of handling intercept: center X and y for dense
        self.X_offset = None
        if fit_intercept:
            X, y, X_offset, y_offset, _, _ = _preprocess_data(
                X=X, y=y, fit_intercept=fit_intercept, copy=True
            )
            self.X_offset = X_offset
            self.y_offset = y_offset

        self.X, self.y = X, y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        _, self.coefs, _ = celer_path(
            self.X,
            self.y,
            pb="lasso",
            alphas=self.lambdas / len(self.y),
            prune=1,
            tol=1e-15,
            max_iter=n_iter,
            max_epochs=100_000,
            X_offset=None,  # self.X is already centered
            X_scale=None,  # self.X is already centered
            return_thetas=False,
        )

        if self.fit_intercept:
            intercept = self.y_offset - self.X_offset @ self.coefs
            self.coefs = np.vstack([self.coefs, intercept])

    def get_result(self):
        return dict(coefs=self.coefs)
