import warnings

from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from numpy import vstack
    from scipy import sparse
    from celer import celer_path
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model._base import _preprocess_data


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
        # celer/sklearn way of handling intercept: center X and y for dense
        if fit_intercept:
            X, y, X_offset, y_offset, _ = _preprocess_data(
                X, y, fit_intercept, return_mean=True
            )
            self.X_offset = X_offset
            self.y_offset = y_offset

        self.X, self.y = X, y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

    def run(self, tol):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        if self.fit_intercept and sparse.issparse(self.X):
            _, self.coefs, _ = celer_path(
                self.X,
                self.y,
                pb="lasso",
                alphas=self.lambdas / len(self.y),
                prune=1,
                tol=tol,
                max_iter=1_000,
                max_epochs=100_000,
                X_offset=self.X_offset,
            )
        else:
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

        if self.fit_intercept:
            intercept = self.y_offset - self.X_offset @ self.coefs
            self.coefs = vstack((self.coefs, intercept))

    def get_result(self):
        return self.coefs
