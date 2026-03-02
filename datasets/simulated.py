from benchopt import BaseDataset, safe_import_context
from benchopt.datasets.simulated import make_correlated_data

with safe_import_context() as import_ctx:
    from benchmark_utils.utils import preprocess_data


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        "n_samples, n_features, n_signals": [
            (10_000, 200, 20),
            (200, 10_000, 20),
        ],
        "rho": [0, 0.5],
    }

    def get_data(self):
        # The seed varies when changing parameter in the dataset, objective or
        # when running multiple repetitions, but is the same across different
        # solvers.
        seed = self.get_seed(
            use_dataset=True, use_repetition=True, use_objective=True
        )
        X, y, _ = make_correlated_data(
            self.n_samples,
            self.n_features,
            rho=self.rho,
            density=self.n_signals / self.n_features,
            random_state=seed,
        )

        X, y = preprocess_data(X, y)

        return dict(X=X, y=y)
