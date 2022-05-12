import numpy as np
from benchopt import BaseObjective, safe_import_context
from numpy.linalg import norm

with safe_import_context() as import_ctx:
    from sklearn.linear_model import Lasso


class Objective(BaseObjective):
    name = "Lasso Path"

    parameters = {
        "fit_intercept": [True, False],
        "n_lambda": [100],
    }

    def __init__(self, fit_intercept=False, n_lambda=100, lambda_min_ratio=None):
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

        # the following code fits the path once using the same path-stopping
        # criteria that glmnet uses to find a reasonable lambda path in order
        # to obtain a reasonable lambda sequence
        if self.lambda_min_ratio is None:
            self.lambda_min_ratio = 1e-2 if self.n_samples < self.n_features else 1e-4

        lambda_max = self._get_lambda_max()
        lambdas = np.logspace(
            np.log(lambda_max),
            np.log(lambda_max * self.lambda_min_ratio),
            num=self.n_lambda,
            base=np.exp(1),
        )

        # the pmax setting is redundant because dfmax is hardcoded, but it's
        # kept just to be explicit about how this is done
        dfmax = self.n_features + 1
        pmax = min(dfmax * 2 + 20, self.n_features)

        lasso = Lasso(
            max_iter=10000,
            alpha=lambda_max / self.n_samples,
            # TODO(jolars): consider whether tol might be too high
            tol=1e-4,
            fit_intercept=self.fit_intercept,
            warm_start=True,
        )

        r = y - np.mean(y) if self.fit_intercept else y.copy()
        null_dev = 0.5 * np.linalg.norm(r) ** 2
        dev_prev = null_dev

        i = 0
        ever_active = np.array([])

        for i, lam in enumerate(lambdas):
            lasso.alpha = lam / self.n_samples
            lasso.fit(X, y)

            w = lasso.coef_
            intercept = lasso.intercept_

            dev = 0.5 * np.linalg.norm(y - X @ w - intercept) ** 2

            dev_ratio = 1 - dev / null_dev
            dev_change = 1 - dev / dev_prev
            dev_prev = dev

            n_nonzero = np.sum(w != 0)
            ever_active = np.union1d(ever_active, np.where(w != 0))

            if (
                n_nonzero >= dfmax
                or dev_ratio > 0.999
                or (i > 0 and dev_change < 1e-5)
                or len(ever_active) >= pmax
            ):
                break

        self.lambdas = lambdas[: i + 1]

    def get_one_solution(self):
        return np.zeros([self.n_features, len(self.lambdas)])

    def compute(self, coefs):
        intercepts = []
        if self.fit_intercept:
            betas = coefs[: self.n_features, :]
            intercepts = coefs[-1, :]
        else:
            betas = coefs

        path_length = len(self.lambdas)

        primals = np.empty(path_length, dtype=np.float64)
        duals = np.empty(path_length, dtype=np.float64)

        for i in range(path_length):
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

    def to_dict(self):
        return dict(
            X=self.X,
            y=self.y,
            lambdas=self.lambdas,
            fit_intercept=self.fit_intercept,
        )
