import warnings

from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
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
            if sparse.issparse(X):
                #  X, y, X_offset, y_offset, X_scale =
                X, y, _, _, _ = _preprocess_data(X, y, fit_intercept)
            else:
                X, y, _, _, _ = _preprocess_data(X, y, fit_intercept, copy=True)
        # center y always (readd y_offset to intercept after fit)
        # dense case: center X, substract X_offset @ coefs to intercept,
        # scale coefs by X_scale
        # sparse case: cannot modify X, pass X_offset and X_scale to celer_path
        # which which handle them in the computation. Modify coef as in dense case.

        self.X, self.y = X, y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

    def run(self, tol):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        if self.fit_intercept:

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

    def get_result(self):
        # coefs /= X_scale
        # intercept = y_offset - np.dot(X_offset, coefs)
        return self.coefs
