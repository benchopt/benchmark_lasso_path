from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from scipy import sparse
    from gsroptim.lasso import lasso_path


class Solver(BaseSolver):
    name =  "gsroptim"
    stopping_strategy = "iteration"

    install_cmd = "conda"
    requirements = [
        'pip:git+https://github.com/EugeneNdiaye/Gap_Safe_Rules.git@master'
    ]
    references = [
        "E. Ndiaye, O. Fercoq, A. Gramfort and J. Salmon, JMLR, "
        '"Gap Safe screening rules for sparsity enforcing penalties", '
        "vol. 18, pp. 1-33 (2017)"
    ]

    def skip(self, X, y, lambdas, fit_intercept):
        if fit_intercept and sparse.issparse(X):
            return (
                True,
                f"{self.name} doesn't handle fit_intercept with sparse data",
            )

        return False, None

    def set_objective(self, X, y, lambdas, fit_intercept):
        self.X, self.y = X, y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        self.coefs = lasso_path(self.X, self.y, self.lambdas, eps=1e-12,
                                max_iter=n_iter, fit_intercept=self.fit_intercept)[1]

    @staticmethod
    def get_next(previous):
        return previous + 1

    def get_result(self):
        return self.coefs
