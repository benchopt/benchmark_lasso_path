from benchopt import BaseSolver, safe_import_context
from benchopt.runner import INFINITY
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.helpers.r_lang import import_rpackages
    from rpy2 import robjects
    from rpy2.robjects import numpy2ri, packages
    from scipy import sparse

    # Setup the system to allow rpy2 running
    numpy2ri.activate()
    import_rpackages("glmnet")


class Solver(BaseSolver):
    name = "glmnet"

    install_cmd = "conda"
    requirements = ["r-base", "rpy2", "r-glmnet", "r-matrix"]
    references = [
        'J. Friedman, T. J. Hastie and R. Tibshirani, "Regularization paths '
        'for generalized linear models via coordinate descent", '
        "J. Stat. Softw., vol. 33, no. 1, pp. 1-22, NIH Public Access (2010)"
    ]
    support_sparse = True

    # We use the tolerance strategy because if maxit is too low and glmnet
    # convergence check fails, it returns an empty model
    stopping_criterion = SufficientProgressCriterion(
        patience=7, eps=1e-38, strategy="tolerance"
    )

    def set_objective(self, X, y, lambdas, fit_intercept, n_lambda, lambda_min_ratio):
        self.n, self.p = X.shape
        self.n_lambda = n_lambda
        if sparse.issparse(X):
            r_Matrix = packages.importr("Matrix")
            X = X.tocoo()
            self.X = r_Matrix.sparseMatrix(
                i=robjects.IntVector(X.row + 1),
                j=robjects.IntVector(X.col + 1),
                x=robjects.FloatVector(X.data),
                dims=robjects.IntVector(X.shape),
            )
        else:
            self.X = robjects.r.matrix(X, X.shape[0], X.shape[1])
        self.y = robjects.FloatVector(y)
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

        self.glmnet = robjects.r["glmnet"]

    def run(self, tol):
        # Even if maxit=0, glmnet can return non zero coefficients. To get the
        # initial point on the curve, we manually return the intercept-only
        # solution
        if tol == INFINITY:
            self.coefs = np.zeros((self.p, self.n_lambda))
            if self.fit_intercept:
                intercepts = np.empty(self.n_lambda)
                intercepts.fill(np.mean(np.array(self.y)))
                self.coefs = np.vstack((self.coefs, intercepts))
        else:
            # we need thresh to decay fast, otherwise the objective curve can
            # plateau before convergence
            thresh = tol**1.8

            fit_dict = {"lambda": self.lambdas / len(self.y)}

            glmnet_fit = self.glmnet(
                self.X,
                self.y,
                intercept=self.fit_intercept,
                standardize=False,
                maxit=1_000_000,
                thresh=thresh,
                **fit_dict,
            )
            results = dict(zip(glmnet_fit.names, list(glmnet_fit)))
            as_matrix = robjects.r["as"]
            self.coefs = np.array(as_matrix(results["beta"], "matrix"))

            if self.fit_intercept:
                self.coefs = np.vstack((self.coefs, results["a0"]))

    def get_result(self):
        return self.coefs
