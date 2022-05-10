from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_openml


class Dataset(BaseDataset):

    name = "flora"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    def get_data(self):
        X, y = fetch_openml(data_id=42708, return_X_y=True)
        return dict(X=X, y=y)
