from benchopt import BaseObjective
import numpy as np
from numpy.linalg import norm

from benchmark_utils.utils import select_lambdas


class Objective(BaseObjective):
    """Lasso Path - L1 regularized linear regression."""
    min_benchopt_version = "1.8.2"
    name = "Lasso Path"

    install_cmd = "conda"
    requirements = [
        "r-base",
        "rpy2>=3.6.0",
        "r-glmnet>=4.0",
        "r-matrix",
        "scikit-learn>=1.7.0",
    ]

    parameters = {
        "fit_intercept": [True, False],
        "n_lambda": [100],
    }

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape
        self.lambdas = select_lambdas(X, y, self.fit_intercept)

    def get_one_result(self):
        output_shape = self.n_features + int(self.fit_intercept)
        return dict(coefs=np.zeros([output_shape, len(self.lambdas)]))

    def evaluate_result(self, coefs):
        if self.fit_intercept:
            betas = coefs[: self.n_features, :]
            intercepts = coefs[-1, :]
        else:
            betas = coefs
            intercepts = np.zeros(self.n_features)

        path_length = len(self.lambdas)

        primals = np.empty(path_length, dtype=np.float64)
        duals = np.empty(path_length, dtype=np.float64)

        for i in range(path_length):
            beta = betas[:, i]

            residual = self.y - self.X @ beta - intercepts[i]

            dual_scale = max(1, norm(self.X.T @ residual, ord=np.inf) / self.lambdas[i])

            primals[i] = 0.5 * norm(residual) ** 2 + self.lambdas[i] * norm(beta, 1)
            duals[i] = (
                0.5 * norm(self.y) ** 2
                - 0.5 * norm(self.y - residual / dual_scale) ** 2
            )

        gaps = primals - duals

        max_rel_duality_gap = np.max(gaps / primals)
        max_abs_duality_gap = np.max(gaps)
        mean_rel_duality_gaps = np.mean(gaps / primals)
        mean_abs_duality_gaps = np.mean(gaps)

        return dict(
            value=np.sum(primals),
            max_rel_duality_gap=max_rel_duality_gap,
            max_abs_duality_gap=max_abs_duality_gap,
            mean_rel_duality_gaps=mean_rel_duality_gaps,
            mean_abs_duality_gaps=mean_abs_duality_gaps,
        )

    def get_objective(self):
        return dict(
            X=self.X,
            y=self.y,
            lambdas=self.lambdas,
            fit_intercept=self.fit_intercept,
        )
