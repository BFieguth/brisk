from unittest import mock
from importlib import util

import numpy as np
import pandas as pd
import pytest
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from brisk.services.utility import UtilityService
from brisk.configuration.algorithm_wrapper import AlgorithmWrapper
from brisk.configuration.algorithm_collection import AlgorithmCollection
from brisk.services import get_services


@pytest.fixture
def mock_services(mock_brisk_project):
    return get_services()

@pytest.fixture
def algorithm_config():
    """Create a mock algorithm configuration."""
    wrapper1 = AlgorithmWrapper(
        name="logistic_regression",
        display_name="Logistic Regression",
        algorithm_class=LogisticRegression,
        default_params={"max_iter": 1000}
    )
    wrapper2 = AlgorithmWrapper(
        name="random_forest",
        display_name="Random Forest Classifier",
        algorithm_class=RandomForestClassifier,
        default_params={"n_estimators": 100}
    )
    return AlgorithmCollection(wrapper1, wrapper2)


@pytest.fixture
def group_index_train():
    """Create mock group indices for training data."""
    return {
        "indices": np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        "group_names": ["group_a", "group_b", "group_c", "group_d"]
    }


@pytest.fixture
def group_index_test():
    """Create mock group indices for test data."""
    return {
        "indices": np.array([0, 1, 2, 3]),
        "group_names": ["group_a", "group_b", "group_c", "group_d"]
    }


@pytest.fixture
def utility_service_with_groups(group_index_train, group_index_test):
    return UtilityService(
        name="utility",
        group_index_train=group_index_train,
        group_index_test=group_index_test
    )


@pytest.fixture
def utility_service_no_groups():
    return UtilityService(
        name="utility",
        group_index_train=None,
        group_index_test=None
    )


@pytest.fixture
def data_manager(mock_brisk_project, tmp_path, mock_services):
    with mock.patch("brisk.data.data_manager.get_services", return_value=mock_services):
        data_file = tmp_path / "data.py"
        spec = util.spec_from_file_location("data", data_file)
        data_module = util.module_from_spec(spec)
        spec.loader.exec_module(data_module)
        return data_module.BASE_DATA_MANAGER


@pytest.fixture
def regression100_data(data_manager, tmp_path, mock_services):
    with mock.patch("brisk.data.data_split_info.get_services", return_value=mock_services):
        data_file = tmp_path / "datasets" / "regression100.csv"
        splits = data_manager.split(
            data_file,
            categorical_features=None,
            group_name="test_group",
            filename="regression100"
        )
        split = splits.get_split(0)
        X_train, y_train = split.get_train()
        X_train.attrs["is_test"] = False
        y_train.attrs["is_test"] = False
        # Fixed predictions for testing
        predictions = pd.Series([
            -2.21217924e+02,  1.65825828e+01,  3.11337997e+00,  4.82845707e+00,
            1.54536773e+02,  8.44781940e+00, -1.80027578e+01, -9.21100836e+00,
            -7.66634881e+01,  4.97458872e+01, -3.05020000e+01, -3.55248800e+00,
            2.94593309e+00, -1.11133601e+02, -8.14443112e+01, -1.52532226e+02,
            2.10043321e+02, -6.84191478e+01, -1.34531149e+01, -1.66022008e+01,
            4.28049122e+01,  2.71563536e+01,  1.72066306e+01, -2.29122007e+01,
            3.26928949e+00, -3.88841245e+01, -2.51496666e+01,  4.57463424e+01,
            2.09237656e+02, -1.08703023e+02,  1.10276525e+02,  5.05850856e+01,
            -1.04817919e+02,  9.56853195e+01, -5.94308228e+01, -1.52260781e+01,
            6.53448761e+01,  5.78285064e+01,  6.13106332e+01,  6.17112832e+00,
            2.00254563e+01, -5.84401526e+01, -2.52671245e+02,  2.78899119e+01,
            1.39359311e+01, -1.92903666e+02,  1.42964142e+02,  2.18549442e+01,
            1.08853746e+02,  1.46647263e+00, -3.10520667e+00,  2.28175081e+00,
            -1.40259801e+02, -2.59084174e+01, -5.10372303e+01,  1.81746487e+01,
            1.54168651e+01, -1.46359330e+01, -5.71988356e+01,  6.53202893e+00,
            -1.36070162e+02,  2.98898394e+01, -3.83434619e+01,  2.20278571e+02,
            -2.31905202e+01,  7.92614171e-03,  1.24631200e-01, -1.39743453e+02,
            9.85133780e+01, -1.17681824e+02,  2.33802488e+01,  8.19178834e+00,
            9.77163755e+01,  9.20506388e+01,  4.95536575e+01, -4.81291179e+00,
            1.34413981e+01, -3.55054901e-01,  7.70581565e+01, -7.68183871e+01
        ])
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True), predictions


class TestUtilityService:
    def test_init_with_groups(self, utility_service_with_groups, group_index_train, group_index_test):
        assert utility_service_with_groups.name == "utility"
        assert utility_service_with_groups.algorithm_config is None
        assert utility_service_with_groups.group_index_train == group_index_train
        assert utility_service_with_groups.group_index_test == group_index_test
        assert utility_service_with_groups.data_has_groups is True

    def test_init_without_groups(self, utility_service_no_groups):
        assert utility_service_no_groups.name == "utility"
        assert utility_service_no_groups.algorithm_config is None
        assert utility_service_no_groups.group_index_train is None
        assert utility_service_no_groups.group_index_test is None
        assert utility_service_no_groups.data_has_groups is False

    def test_set_split_indices_with_groups(self, utility_service_no_groups, group_index_train, group_index_test):
        utility_service_no_groups.set_split_indices(group_index_train, group_index_test)
        
        assert utility_service_no_groups.group_index_train == group_index_train
        assert utility_service_no_groups.group_index_test == group_index_test
        assert utility_service_no_groups.data_has_groups is True

    def test_set_split_indices_without_groups(self, utility_service_with_groups):
        utility_service_with_groups.set_split_indices(None, None)
        
        assert utility_service_with_groups.group_index_train is None
        assert utility_service_with_groups.group_index_test is None
        assert utility_service_with_groups.data_has_groups is False

    def test_set_split_indices_partial_groups(self, utility_service_no_groups, group_index_train):
        # Only training groups, no test groups
        utility_service_no_groups.set_split_indices(group_index_train, None)
        
        assert utility_service_no_groups.group_index_train == group_index_train
        assert utility_service_no_groups.group_index_test is None
        assert utility_service_no_groups.data_has_groups is False

    def test_get_algo_wrapper(self, utility_service_with_groups, algorithm_config):
        utility_service_with_groups.set_algorithm_config(algorithm_config)
        wrapper = utility_service_with_groups.get_algo_wrapper("logistic_regression")
        
        assert wrapper.name == "logistic_regression"
        assert wrapper.display_name == "Logistic Regression"
        assert wrapper.algorithm_class == LogisticRegression

    def test_get_group_index_train(self, utility_service_with_groups, group_index_train):
        result = utility_service_with_groups.get_group_index(is_test=False)
        assert result == group_index_train

    def test_get_group_index_test(self, utility_service_with_groups, group_index_test):
        result = utility_service_with_groups.get_group_index(is_test=True)
        assert result == group_index_test

    def test_get_group_index_no_groups(self, utility_service_no_groups):
        result_train = utility_service_no_groups.get_group_index(is_test=False)
        result_test = utility_service_no_groups.get_group_index(is_test=True)
        
        assert result_train is None
        assert result_test is None

    def test_get_cv_splitter(self, regression100_data, utility_service_no_groups, utility_service_with_groups):
        utility_service_no_groups._other_services["logging"] = mock.MagicMock()
        utility_service_with_groups._other_services["logging"] = mock.MagicMock()
        _, y, _ = regression100_data
        y_categorical = pd.Series([1, 0] * 25) 
        y_categorical.attrs["is_test"] = False

        splitter, _ = utility_service_no_groups.get_cv_splitter(y, 5, None)
        assert isinstance(splitter, model_selection.KFold)

        splitter, _ = utility_service_no_groups.get_cv_splitter(y, 5, 2)
        assert isinstance(splitter, model_selection.RepeatedKFold)

        splitter, _ = utility_service_no_groups.get_cv_splitter(
            y_categorical, 5, None
        )
        assert isinstance(splitter, model_selection.StratifiedKFold)

        splitter, _ = utility_service_no_groups.get_cv_splitter(
            y_categorical, 5, 2
        )
        assert isinstance(splitter, model_selection.RepeatedStratifiedKFold)

        utility_service_with_groups.data_has_groups = True
        utility_service_with_groups.group_index_train = {
            "values": np.array([1, 2, 3]),
            "indices": np.array([0, 1, 2]),
            "series": pd.Series([1, 2, 3])
        }
        utility_service_with_groups.group_index_test = {
            "values": np.array([4, 5, 6]),
            "indices": np.array([3, 4, 5]),
            "series": pd.Series([4, 5, 6])
        }
        splitter, _ = utility_service_with_groups.get_cv_splitter(y, 5, None)
        assert isinstance(splitter, model_selection.GroupKFold)

        splitter, _ = utility_service_with_groups.get_cv_splitter(y, 5, 2)
        assert isinstance(splitter, model_selection.GroupKFold)

        splitter, _ = utility_service_with_groups.get_cv_splitter(
            y_categorical, 5, None
        )
        assert isinstance(splitter, model_selection.StratifiedGroupKFold)

        splitter, _ = utility_service_with_groups.get_cv_splitter(
            y_categorical, 5, 2
        )
        assert isinstance(splitter, model_selection.StratifiedGroupKFold)

    def test_categorical_detection_threshold(self, utility_service_no_groups):
        # Create data where unique values / total length = exactly 5%
        y = pd.Series([0] * 95 + [1] * 5, name="target")
        y.attrs = {"is_test": False}
        
        splitter, _ = utility_service_no_groups.get_cv_splitter(y, cv=3)
        assert isinstance(splitter, model_selection.StratifiedKFold)
        
        # Create data where unique values / total length < 5%
        y = pd.Series(list(range(100)), name="target")
        y.attrs = {"is_test": False}
        
        splitter, _ = utility_service_no_groups.get_cv_splitter(y, cv=3)
        assert isinstance(splitter, model_selection.KFold)

    def test_get_group_index_uses_correct_attrs(self, utility_service_with_groups):
        y_train = pd.Series([1, 2, 3], name="target")
        y_train.attrs = {"is_test": False}
        
        y_test = pd.Series([4, 5, 6], name="target")
        y_test.attrs = {"is_test": True}
        
        with mock.patch.object(
            utility_service_with_groups,
            'get_group_index',
            wraps=utility_service_with_groups.get_group_index
        ) as mock_get_group:
            utility_service_with_groups.get_cv_splitter(y_train, cv=3)
            mock_get_group.assert_called_with(False)
            
            utility_service_with_groups.get_cv_splitter(y_test, cv=3)
            mock_get_group.assert_called_with(True)
