import warnings

from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import lasso_path


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

    def set_objective(self, X, y, lambdas, fit_intercept, n_lambda, lambda_min_ratio):
        self.X = X
        self.y = y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept
        self.n_lambda = n_lambda

    def skip(self, X, y, lambdas, fit_intercept, n_lambda, lambda_min_ratio):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def run(self, n_iter):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        _, self.coefs, _ = lasso_path(
            self.X,
            self.y,
            alphas=self.lambdas / len(self.y),
            max_iter=n_iter,
            tol=1e-35,
        )

    def get_result(self):
        beta = self.coefs
        return beta
