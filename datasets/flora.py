from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from openml.datasets import get_dataset
    from scipy.sparse import csc_array


class Dataset(BaseDataset):

    name = "flora"

    install_cmd = "conda"
    requirements = ["openml", "scipy"]

    def get_data(self):
        dataset = get_dataset(42708)
        X, y, _, _ = dataset.get_data("target")

        X = csc_array(X, dtype="d")
        y = np.array(y)

        return dict(X=X, y=y)
