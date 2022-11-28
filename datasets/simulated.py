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

    def __init__(
        self, n_samples=10, n_features=50, n_signals=5, rho=0, random_state=27
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_signals = n_signals
        self.random_state = random_state
        self.rho = rho

    def get_data(self):
        X, y, _ = make_correlated_data(
            self.n_samples,
            self.n_features,
            rho=self.rho,
            density=self.n_signals / self.n_features,
            random_state=self.random_state,
        )

        X, y = preprocess_data(X, y)

        return dict(X=X, y=y)
