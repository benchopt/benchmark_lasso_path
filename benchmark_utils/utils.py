from benchopt import safe_import_context

# with safe_import_context() as import_ctx:
import numpy as np
from benchopt.helpers.r_lang import import_rpackages
from rpy2 import robjects
from rpy2.robjects import numpy2ri, packages
from scipy import sparse
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

    # Setup the system to allow rpy2 running
numpy2ri.activate()
import_rpackages("glmnet")


def preprocess_data(X, y=None):
    X = VarianceThreshold().fit_transform(X)

    if sparse.issparse(X):
        X = MaxAbsScaler().fit_transform(X).tocsc()
    else:
        X = StandardScaler().fit_transform(X)

    return X, y


def select_lambdas(X, y, fit_intercept):
    # run glmnet once to obtain a lambda sequence
    if sparse.issparse(X):
        r_Matrix = packages.importr("Matrix")
        X_coo = X.tocoo()
        X_tmp = r_Matrix.sparseMatrix(
            i=robjects.IntVector(X_coo.row + 1),
            j=robjects.IntVector(X_coo.col + 1),
            x=robjects.FloatVector(X_coo.data),
            dims=robjects.IntVector(X_coo.shape),
        )
    else:
        X_tmp = robjects.r.matrix(X, X.shape[0], X.shape[1])

    y_tmp = robjects.FloatVector(y)
    glmnet = robjects.r["glmnet"]
    glmnet_fit = glmnet(
        X_tmp,
        y_tmp,
        intercept=fit_intercept,
        standardize=False,
        maxit=1_000_000,
    )
    results = dict(zip(glmnet_fit.names, list(glmnet_fit)))

    return np.array(results["lambda"]) * len(y)
