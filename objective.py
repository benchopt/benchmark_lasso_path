import numpy as np
from benchopt import BaseObjective
from numpy.linalg import norm


class Objective(BaseObjective):
    name = "Lasso Path"

    parameters = {
        "fit_intercept": [True, False],
        "n_lambda": [100],
    }

    # TODO(jolars): let `lambda_min_ratio` be decided based on n and p
    def __init__(self, fit_intercept=False, n_lambda=100, lambda_min_ratio=1e-4):
        self.fit_intercept = fit_intercept
        self.lambda_min_ratio = lambda_min_ratio
        self.n_lambda = n_lambda

    def _get_lambda_max(self):
        if self.fit_intercept:
            return abs(self.X.T @ (self.y - self.y.mean())).max()
        else:
            return abs(self.X.T.dot(self.y)).max()

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape

        if self.lambda_min_ratio is None:
            self.lambda_min_ratio = 1e-4 if self.n_samples < self.n_features else 1e-6

        lambda_max = self._get_lambda_max()
        self.lambdas = np.logspace(
            np.log(lambda_max),
            np.log(lambda_max * self.lambda_min_ratio),
            num=self.n_lambda,
            base=np.exp(1),
        )

    def compute(self, coefs):
        intercepts = np.empty(self.n_lambda)
        if self.fit_intercept:
            betas = coefs[: self.n_features, :]
            intercepts = coefs[-1, :]
        else:
            betas = coefs

        primals = np.empty(self.n_lambda)
        duals = np.empty(self.n_lambda)

        for i in range(self.n_lambda):
            beta = betas[:, i]

            residual = self.y - self.X @ beta
            if self.fit_intercept:
                residual -= intercepts[i]

            dual_scale = max(1, norm(self.X.T @ residual, ord=np.inf) / self.lambdas[i])

            primals[i] = 0.5 * norm(residual) ** 2 + self.lambdas[i] * norm(beta, 1)
            duals[i] = (
                0.5 * norm(self.y) ** 2
                - 0.5 * norm(self.y - residual / dual_scale) ** 2
            )

        max_rel_duality_gap = np.max((primals - duals) / primals)
        sum_duality_gaps = np.sum(primals - duals)

        return dict(
            value=np.sum(primals),
            max_rel_duality_gap=max_rel_duality_gap,
            sum_duality_gaps=sum_duality_gaps,
        )

    def to_dict(self):
        return dict(
            X=self.X,
            y=self.y,
            lambdas=self.lambdas,
            fit_intercept=self.fit_intercept,
            n_lambda=self.n_lambda,
            lambda_min_ratio=self.lambda_min_ratio,
        )
