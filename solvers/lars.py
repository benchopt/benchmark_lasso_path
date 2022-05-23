from benchopt import BaseSolver, safe_import_context
from benchopt.runner import INFINITY
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import warnings

    import numpy as np
    import scipy.sparse as sparse
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import lars_path


class Solver(BaseSolver):
    name = "LARS"
    support_sparse = False
    # We use a tolerance progress criterion and patience=1 so that we stop
    # after we've solved the path since LARS solves the path analytically.
    stopping_criterion = SufficientProgressCriterion(
        patience=1, eps=1e-10, strategy="tolerance"
    )

    install_cmd = "conda"
    requirements = ["scikit-learn"]
    references = [
        "B. Efron, T. Hastie, I. Johnstone, R. Tibshirani"
        '"Least Angle Regression", Annals of Statistics, '
        " vol. 32 (2), pp. 407-499 (2004)"
    ]

    def set_objective(self, X, y, lambdas, fit_intercept):
        if fit_intercept and not sparse.issparse(X):
            self.y_offset = np.mean(y)
            self.X_offset = np.mean(X, axis=0)
            X = X - self.X_offset
            y = y - self.y_offset

        self.X = X
        self.y = y

        self.n, self.p = X.shape
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

    def skip(self, X, y, lambdas, fit_intercept):
        if fit_intercept and sparse.issparse(X):
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def run(self, tol):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # NOTE: For the first tolerance we just return a path of zeroes. For
        # all other input we solve the complete path since the notion of
        # tolerance or iterations does not make sense for LARS.
        if tol == INFINITY:
            self.coefs = np.zeros((self.p, len(self.lambdas)))
            self.first_run = True
        else:
            self.alphas, _, self.coefs = lars_path(
                self.X,
                self.y,
                alpha_min=self.lambdas[-1] / len(self.y),
                max_iter=np.iinfo(int).max,
                method="lasso",
                return_path=True,
            )
            self.first_run = False

        if self.fit_intercept and not sparse.issparse(self.X):
            intercepts = np.zeros(self.coefs.shape[1])
            if tol != INFINITY:
                for i in range(self.coefs.shape[1]):
                    intercepts[i] = self.y_offset - self.X_offset @ self.coefs[:, i]

            self.coefs = np.vstack((self.coefs, intercepts))

    def get_result(self):
        # XXX TO IMPROVE: find coeficients at regularization in self.lambdas
        # coefs returned by lasso_path are sampled at the nodes of the path
        # coefficients between two nodes can be found by linear interpolation

        # Getting values at lambdas grid via interpolation done coordinate
        # by coordinate with np.interp.
        # There should be a more efficient way of doing this.
        # Also np.interp requires increasing inputs,
        # so we need to reverse all arrays
        if self.first_run:
            coefs = self.coefs
        else:
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
