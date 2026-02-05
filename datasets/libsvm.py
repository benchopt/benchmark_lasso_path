from benchopt import BaseDataset
from libsvmdata import fetch_libsvm

from benchmark_utils.utils import preprocess_data


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        "dataset":
            [
                "bodyfat",  # 252 samples, 14 features
                "finance",  # E2006-log1p, 16,087 + 3,308 samples, 4,272,227 features
                "finance-tf-idf",  # E2006-tfidf, 16,087 + 3,308 smpls, 150,360 ftrs
                "YearPredictionMSD",  # 463,715 + 51,630 samples, 90 features
            ],
    }

    install_cmd = "conda"
    requirements = ["libsvmdata"]

    def __init__(self, dataset="bodyfat"):
        self.dataset = dataset
        self.X, self.y = None, None

    def get_data(self):
        X, y = fetch_libsvm(self.dataset)
        X, y = preprocess_data(X, y)

        return dict(X=X, y=y)
