from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import warnings

    from gsroptim.lasso import lasso_path
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "gsroptim"
    stopping_strategy = "iteration"

    install_cmd = "conda"
    requirements = ["pip:git+https://github.com/EugeneNdiaye/Gap_Safe_Rules.git@master"]
    references = [
        "E. Ndiaye, O. Fercoq, A. Gramfort and J. Salmon, JMLR, "
        '"Gap Safe screening rules for sparsity enforcing penalties", '
        "vol. 18, pp. 1-33 (2017)"
    ]

    def set_objective(self, X, y, lambdas, fit_intercept):
        self.X, self.y = X, y
        self.lambdas = lambdas
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        self.coefs = lasso_path(
            self.X,
            self.y,
            self.lambdas,
            eps=1e-12,
            max_iter=n_iter,
            fit_intercept=self.fit_intercept,
        )[1]

    def get_result(self):
        return self.coefs
