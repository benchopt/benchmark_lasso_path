from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelBinarizer

    preprocess_data = import_ctx.import_from("utils", "preprocess_data")


class Dataset(BaseDataset):

    name = "leukemia"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    def get_data(self):
        # Unlike libsvm[leukemia], this dataset corresponds to the whole
        # leukemia  data with train + test data (72 samples) and not just
        # the training set.
        X, y = fetch_openml("leukemia", return_X_y=True)
        X = X.to_numpy()
        y = LabelBinarizer().fit_transform(y)[:, 0].astype(X.dtype)

        X, y = preprocess_data(X, y)

        return dict(X=X, y=y)
