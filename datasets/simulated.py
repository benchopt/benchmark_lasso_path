from benchopt import BaseDataset
from benchopt.datasets.simulated import make_correlated_data


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        "n_samples, n_features": [(5_000, 200), (200, 5_000)],
        "rho": [0, 0.5],
    }

    def __init__(self, n_samples=10, n_features=50, rho=0, random_state=27):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.rho = rho

    def get_data(self):
        X, y, _ = make_correlated_data(
            self.n_samples,
            self.n_features,
            rho=self.rho,
            random_state=self.random_state,
        )

        data = dict(X=X, y=y)

        return self.n_features, data
