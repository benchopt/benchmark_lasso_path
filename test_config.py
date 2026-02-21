import pytest


def check_test_dataset_get_data(benchmark, dataset_class):
    """Hook to skip `test_dataset_get_data` for specific datasets."""
    if dataset_class.name == "breheny":
        pytest.xfail(
            "breheny dataset is not available anymore on S3--gives 403 error."
        )


def check_test_solver_install(benchmark, solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    if solver_class.name == "gsroptim":
        pytest.xfail(
            "gsroptim is not compatible with pip 26.0+."
        )
