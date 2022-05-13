import warnings

from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import lasso_path
    from sklearn.linear_model._base import _preprocess_data


class Solver(BaseSolver):
    name = "sklearn"

    install_cmd = "conda"
    requirements = ["scikit-learn"]
    references = [
        "F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, "
        "O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, "
        "J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot"
        " and E. Duchesnay"
        '"Scikit-learn: Machine Learning in Python", J. Mach. Learn. Res., '
        "vol. 12, pp. 2825-283 (2011)"
    ]

    def set_objective(self, X, y, lambdas, fit_intercept):
        # sklearn way of handling intercept: center y and X.
        # When X is sparse, it is not centered in order not to break sparsity
        if fit_intercept:
            X, y, X_offset, y_offset, _ = _preprocess_data(
                X, y, fit_intercept, return_mean=True, copy=True,
            )
            self.X_offset = X_offset
            self.y_offset = y_offset

        self.X = X
        self.y = y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        if self.fit_intercept and sparse.issparse(self.X):
            _, self.coefs, _ = lasso_path(
                self.X,
                self.y,
                alphas=self.lambdas / len(self.y),
                max_iter=n_iter,
                tol=1e-35,
                X_offset=self.X_offset,
                X_scale=np.ones_like(self.X_offset),
            )
        else:
            _, self.coefs, _ = lasso_path(
                self.X,
                self.y,
                alphas=self.lambdas / len(self.y),
                max_iter=n_iter,
                tol=1e-35,
            )

        if self.fit_intercept:
            intercepts = self.y_offset - self.X_offset @ self.coefs
            self.coefs = np.vstack((self.coefs, intercepts))

    def get_result(self):
        beta = self.coefs
        return beta
