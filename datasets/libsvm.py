from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm

    preprocess_data = import_ctx.import_from("utils", "preprocess_data")


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        "dataset": ["finance", "finance-tf-idf", "YearPredictionMSD"],
    }

    install_cmd = "conda"
    requirements = ["pip:libsvmdata"]

    def __init__(self, dataset="bodyfat"):
        self.dataset = dataset
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            self.X, self.y = fetch_libsvm(self.dataset)

        self.X, self.y = preprocess_data(self.X, self.y)

        return dict(X=self.X, y=self.y)
