"""Test builtin evaluators.

These tests focus on the core logic methods (_generate_plot_data, _create_plot, 
_calculate_measures, etc) and reuse existing test logic from the old evaluation manager.
"""
import pytest
import numpy as np
import pandas as pd
from unittest import mock
from unittest.mock import patch
import importlib.util as util

from brisk.evaluation.evaluators.builtin import (
    common_measures, common_plots, regression_plots, classification_measures,
    classification_plots, dataset_measures, tools
)
from brisk.services.bundle import ServiceBundle
from brisk.services.utility import UtilityService
from brisk.data.data_manager import DataManager
from tests.conftest import get_metric_config, get_algorithm_config


@pytest.fixture
def metric_config():
    return get_metric_config()


@pytest.fixture
def algorithm_config():
    return get_algorithm_config()


@pytest.fixture
def mock_services(algorithm_config):
    services = mock.MagicMock(spec=ServiceBundle)
    services.logger = mock.MagicMock()
    services.logger.logger = mock.MagicMock()
    services.io = mock.MagicMock()
    services.io.output_dir = mock.MagicMock()
    services.utility = UtilityService(
        name="utility",
        algorithm_config=algorithm_config,
        group_index_train=None,
        group_index_test=None
    )
    services.metadata = mock.MagicMock()
    return services


@pytest.fixture
def data_manager(mock_brisk_project, tmp_path):
    data_file = tmp_path / "data.py"
    spec = util.spec_from_file_location("data", data_file)
    data_module = util.module_from_spec(spec)
    spec.loader.exec_module(data_module)
    return data_module.BASE_DATA_MANAGER


@pytest.fixture
def data_manager_group(mock_brisk_project, tmp_path):
    data_manager = DataManager(
        test_size=0.2, 
        n_splits=5,
        split_method="shuffle",
        random_state=42,
        group_column="group"
    )
    return data_manager


@pytest.fixture
def regression_data(data_manager, tmp_path):
    data_file = tmp_path / "datasets" / "regression.csv"
    splits = data_manager.split(
        data_file,
        categorical_features=None,
        table_name=None,
        group_name="test_group",
        filename="regression"
    )
    split = splits.get_split(0)
    X_train, y_train = split.get_train()
    X_train.attrs["is_test"] = False
    y_train.attrs["is_test"] = False
    # Fixed predictions for testing
    predictions = [0.7, 0.23, 0.43, 0.79]
    return X_train, y_train, predictions


@pytest.fixture
def regression100_data(data_manager, tmp_path):
    data_file = tmp_path / "datasets" / "regression100.csv"
    splits = data_manager.split(
        data_file,
        categorical_features=None,
        table_name=None,
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


@pytest.fixture
def regression100_group_data(data_manager_group, tmp_path):
    data_file = tmp_path / "datasets" / "regression100_group.csv"
    splits = data_manager_group.split(
        data_file,
        categorical_features=["cat_feature_1"],
        table_name=None,
        group_name="test_group",
        filename="regression100_group"
    )
    split = splits.get_split(0)
    X_train, y_train = split.get_train()
    X_train.attrs["is_test"] = False
    y_train.attrs["is_test"] = False
    # Fixed predictions for testing
    predictions = pd.Series([
        1.01850213e+01, -7.46359889e+01, -4.09011219e+00, -2.35397907e+01,
        -3.02211452e+01,  1.76412077e+00,  3.97353112e+01,  1.07967132e+01,
        -3.31828396e+01, -1.77577091e+01,  2.56977331e+01, -3.84573232e+01,
        1.53751998e+02,  7.21257811e+01, -1.09582695e+02, -3.72609120e+00,
        -6.25888847e+01,  9.02447204e-02, -4.16125038e+01,  1.55441506e+02,
        6.36313856e+01, -6.13357486e+01, -5.91397966e+01,  1.75016829e+02,
        -3.00998119e+01, -1.25383368e+02, -1.50237347e+01,  2.32111375e+01,
        1.23488947e+02,  6.73223590e+01, -2.87091664e+01,  6.21350354e+01,
        -7.85016362e+01,  8.72685056e+00,  7.88018348e+01, -4.14342288e+00,
        -1.63268607e+02, -2.98423960e+01,  1.23741753e+02, -7.81036802e+01,
        -1.90123755e+01,  1.83399183e+02,  3.04386847e+00,  5.57716151e+01,
        8.71607636e+01,  7.88957428e+01, -1.46311134e+02, -1.02176756e+02,
        2.63408165e+01, -2.16197369e+02,  4.45366036e+01,  1.67651159e+02,
        -2.40110200e+02,  8.04223485e+00,  2.15703834e+01, -1.82211448e+01,
        -7.60577774e+01, -1.82137723e+01, -1.49421749e+01,  9.53903442e+00,
        -3.39756824e+01
    ])
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True), predictions


@pytest.fixture
def classification_data(data_manager, tmp_path):
    data_file = tmp_path / "datasets" / "classification.csv"
    with mock.patch.object(data_manager.services.reporting, 'add_dataset'):
        splits = data_manager.split(
            data_file,
            categorical_features=None,
            table_name=None,
            group_name="test_group",
            filename="classification"
        )
    split = splits.get_split(0)
    X_train, y_train = split.get_train()
    X_train.attrs["is_test"] = False
    y_train.attrs["is_test"] = False
    predictions = pd.Series([
        "A", "B", "A", "B"
    ])
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True), predictions


@pytest.fixture
def classification100_data(data_manager, tmp_path):
    data_file = tmp_path / "datasets" / "classification100.csv"
    splits = data_manager.split(
        data_file,
        categorical_features=["cat_feature_1", "cat_feature_2"],
        table_name=None,
        group_name="test_group",
        filename="classification100"
    )
    split = splits.get_split(0)
    X_train, y_train = split.get_train()
    X_train.attrs["is_test"] = False
    y_train.attrs["is_test"] = False
    predictions = pd.Series([
        0, 0, 0, 1, 1, 1, 0, 1, 1, 0,
        1, 0, 0, 1, 1, 1, 0, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 0, 1, 1, 0,
        0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
        1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 0, 1, 0, 1, 1, 0
    ])
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True), predictions


@pytest.fixture
def classification100_group_data(data_manager_group, tmp_path):
    data_file = tmp_path / "datasets" / "classification100_group.csv"
    splits = data_manager_group.split(
        data_file,
        categorical_features=["cat_feature_1", "cat_feature_2"],
        table_name=None,
        group_name="test_group",
        filename="classification100_group"
    )
    split = splits.get_split(0)
    X_train, y_train = split.get_train()
    X_train.attrs["is_test"] = False
    y_train.attrs["is_test"] = False
    predictions = pd.Series([
        0, 0, 0, 1, 1, 1, 0, 1, 1, 0,
        1, 0, 0, 1, 1, 1, 0, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 0, 1, 1, 0,
        0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
        1
    ])
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True), predictions


@pytest.fixture
def fitted_models(algorithm_config, regression_data):
    """Create fitted models for testing."""
    X, y, _ = regression_data
    
    linear = algorithm_config["linear"].instantiate()
    ridge = algorithm_config["ridge"].instantiate()
    elasticnet = algorithm_config["elasticnet"].instantiate()
    rf = algorithm_config["rf"].instantiate()
    
    linear.fit(X, y)
    ridge.fit(X, y)
    elasticnet.fit(X, y)
    rf.fit(X, y)
    
    # Add wrapper names for identification
    linear.wrapper_name = "linear"
    ridge.wrapper_name = "ridge"
    elasticnet.wrapper_name = "elasticnet"
    rf.wrapper_name = "rf"
    
    return {
        "linear": linear,
        "ridge": ridge,
        "elasticnet": elasticnet,
        "rf": rf
    }


class TestEvaluateModel:
    """Test EvaluateModel evaluator."""
    def test_calculate_measures_regression(self, metric_config, mock_services, regression_data):
        """Test _calculate_measures method with regression.csv."""
        evaluator = common_measures.EvaluateModel(
            "brisk_evaluate_model",
            "Model performance on the specified measures."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        _, y, predictions = regression_data
        metrics = ["mean_absolute_error", "MSE"]
        
        result = evaluator._calculate_measures(predictions, y, metrics)
        
        assert isinstance(result, dict)
        assert np.isclose(result["Mean Absolute Error"], 0.12750)
        assert np.isclose(result["Mean Squared Error"], 0.0189750)

    def test_calculate_measures_regression100(self, metric_config, mock_services, regression100_data):
        """Test _calculate_measures method with regression100.csv."""
        evaluator = common_measures.EvaluateModel(
            "brisk_evaluate_model",
            "Model performance on the specified measures."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        _, y, predictions = regression100_data
        metrics = ["mean_absolute_error", "MAPE", "r2_score"]
        
        result = evaluator._calculate_measures(predictions, y, metrics)
        
        assert isinstance(result, dict)
        assert np.isclose(result["Mean Absolute Error"], 51.98190036274825)
        assert np.isclose(result["Mean Absolute Percentage Error"], 0.4720773856131825)
        assert np.isclose(result["R2 Score"], 0.696868586896576)

    def test_calculate_measures_regression100_group(self, metric_config, mock_services, regression100_group_data):
        """Test _calculate_measures method with regression100_group.csv."""
        evaluator = common_measures.EvaluateModel(
            "brisk_evaluate_model",
            "Model performance on the specified measures."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, predictions = regression100_group_data
        assert X.shape == (61, 5)
        metrics = ["CCC", "MAE"]
        
        result = evaluator._calculate_measures(predictions, y, metrics)
        
        assert isinstance(result, dict)
        assert np.isclose(result["Concordance Correlation Coefficient"], 0.15951165783251642)
        assert np.isclose(result["Mean Absolute Error"], 90.62786907070439)


class TestEvaluateModelCV:
    """Test EvaluateModelCV evaluator."""
    def test_calculate_measures_regression(self, metric_config, mock_services, fitted_models, regression_data):
        """Test _calculate_measures method for cross-validation with regression.csv."""
        evaluator = common_measures.EvaluateModelCV(
            "brisk_evaluate_model_cv",
            "Average model performance across CV splits."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, _ = regression_data
        model = fitted_models["ridge"]
        metrics = ["mean_absolute_error", "MSE"]
        
        result = evaluator._calculate_measures(model, X, y, metrics, cv=2)
        
        assert isinstance(result, dict)
        assert np.isclose(result["Mean Absolute Error"]["mean_score"], -0.6225)
        assert np.isclose(result["Mean Absolute Error"]["std_dev"], 0.09750000000000014)
        for i, score in enumerate(result["Mean Absolute Error"]["all_scores"]):
            assert np.isclose(score, [-0.525, -0.72][i])
        assert np.isclose(result["Mean Squared Error"]["mean_score"], -0.4314625000000001)
        assert np.isclose(result["Mean Squared Error"]["std_dev"], 0.11583750000000012)
        for i, score in enumerate(result["Mean Squared Error"]["all_scores"]):
            assert np.isclose(score, [-0.315625, -0.5473][i])

    def test_calculate_measures_regression100(self, metric_config, mock_services, fitted_models, regression100_data):
        """Test _calculate_measures method with regression100.csv."""
        evaluator = common_measures.EvaluateModelCV(
            "brisk_evaluate_model_cv",
            "Average model performance across CV splits."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, _ = regression100_data
        model = fitted_models["ridge"]
        model.fit(X, y)
        metrics = ["R2", "CCC"]
        
        result = evaluator._calculate_measures(model, X, y, metrics, cv=4)
        
        assert isinstance(result, dict)
        assert np.isclose(result["R2 Score"]["mean_score"], 0.9995030407082244)
        assert np.isclose(result["R2 Score"]["std_dev"], 0.0001309841689887547)
        for i, score in enumerate(result["R2 Score"]["all_scores"]):
            assert np.isclose(score, [0.99945893, 0.99956915, 0.99931588, 0.99966821][i])

    def test_calculate_measures_regression100_group(self, metric_config, mock_services, fitted_models, regression100_group_data):
        """Test _calculate_measures method with regression100_group.csv."""
        evaluator = common_measures.EvaluateModelCV(
            "brisk_evaluate_model_cv",
            "Average model performance across CV splits."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, _ = regression100_group_data
        model = fitted_models["ridge"]
        model.fit(X, y)
        metrics = ["MAE", "MAPE"]
        
        result = evaluator._calculate_measures(model, X, y, metrics, cv=4)
        
        assert isinstance(result, dict)
        assert np.isclose(result["Mean Absolute Error"]["mean_score"], -4.859018024947762)
        assert np.isclose(result["Mean Absolute Error"]["std_dev"], 1.234307892954546)
        for i, score in enumerate(result["Mean Absolute Error"]["all_scores"]):
            assert np.isclose(score, [-3.49985899, -4.3503448, -6.85208558, -4.73378273][i])
    
        assert np.isclose(result["Mean Absolute Percentage Error"]["mean_score"], -0.6175659655526924)
        assert np.isclose(result["Mean Absolute Percentage Error"]["std_dev"], 0.8571714086912385)
        for i, score in enumerate(result["Mean Absolute Percentage Error"]["all_scores"]):
            assert np.isclose(score, [-0.18203515, -2.10087968, -0.08530377, -0.10204527][i])


class TestCompareModels:
    """Test CompareModels evaluator."""
    def test_calculate_measures_regression(self, metric_config, mock_services, fitted_models, regression_data):
        """Test _calculate_measures method for model comparison with regression.csv."""
        evaluator = common_measures.CompareModels(
            "brisk_compare_models",
            "Compare model performance."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, _ = regression_data
        metrics = ["mean_absolute_error", "MSE"]
        
        result = evaluator._calculate_measures(
            fitted_models["linear"], fitted_models["ridge"], fitted_models["elasticnet"],
            X=X, y=y, metrics=metrics, calculate_diff=True
        )
        
        assert isinstance(result, dict)
        assert np.isclose(result["Linear Regression"]["Mean Absolute Error"], 0.2357142857142858)
        assert np.isclose(result["Ridge Regression"]["Mean Absolute Error"], 0.2364864864864865)
        assert np.isclose(result["Elastic Net Regression"]["Mean Absolute Error"], 0.23870030088687683)
        
        assert np.isclose(result["Linear Regression"]["Mean Squared Error"], 0.08642857142857145)
        assert np.isclose(result["Ridge Regression"]["Mean Squared Error"], 0.08651205259313369)
        assert np.isclose(result["Elastic Net Regression"]["Mean Squared Error"], 0.0876768515541036)

        assert np.isclose(result["differences"]["Mean Absolute Error"]["Ridge Regression - Linear Regression"], 0.0007722007722007207)
        assert np.isclose(result["differences"]["Mean Absolute Error"]["Elastic Net Regression - Linear Regression"], 0.0029860151725910333)
        assert np.isclose(result["differences"]["Mean Absolute Error"]["Elastic Net Regression - Ridge Regression"], 0.0022138144003903126)

        assert np.isclose(result["differences"]["Mean Squared Error"]["Ridge Regression - Linear Regression"], 8.34811645622352e-05)
        assert np.isclose(result["differences"]["Mean Squared Error"]["Elastic Net Regression - Linear Regression"], 0.0012482801255321446)
        assert np.isclose(result["differences"]["Mean Squared Error"]["Elastic Net Regression - Ridge Regression"], 0.0011647989609699094)

    def test_calculate_measures_regression100(self, metric_config, mock_services, fitted_models, regression100_data):
        """Test _calculate_measures method with regression100.csv."""
        evaluator = common_measures.CompareModels(
            "brisk_compare_models",
            "Compare model performance."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, _ = regression100_data
        
        # Refit models on larger dataset
        fitted_models["ridge"].fit(X, y)
        fitted_models["linear"].fit(X, y)
        
        metrics = ["mean_absolute_error", "MSE"]
        
        result = evaluator._calculate_measures(
            fitted_models["linear"], fitted_models["ridge"],
            X=X, y=y, metrics=metrics, calculate_diff=True
        )
        
        assert isinstance(result, dict)
        assert np.isclose(result["Linear Regression"]["Mean Absolute Error"], 0.07161930571365704)
        assert np.isclose(result["Ridge Regression"]["Mean Absolute Error"], 1.6613532940107134)
        
        assert np.isclose(result["Linear Regression"]["Mean Squared Error"], 0.008157956114528275)
        assert np.isclose(result["Ridge Regression"]["Mean Squared Error"], 4.036739214414959)

        assert np.isclose(result["differences"]["Mean Absolute Error"]["Ridge Regression - Linear Regression"], 1.5897339882970565)
        assert np.isclose(result["differences"]["Mean Squared Error"]["Ridge Regression - Linear Regression"], 4.028581258300431)


class TestPlotPredVsObs:
    """Test PlotPredVsObs evaluator."""
    def test_generate_plot_data_regression(self, mock_services, regression_data):
        """Test _generate_plot_data method with regression.csv."""
        evaluator = regression_plots.PlotPredVsObs(
            "brisk_plot_pred_vs_obs",
            "Plot predicted vs. observed values."
        )
        evaluator.services = mock_services
        
        _, y, predictions = regression_data
        
        plot_data, max_range = evaluator._generate_plot_data(predictions, y)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert max_range == 1.0
        assert plot_data.shape == (4, 2)

    def test_generate_plot_data_regression100(self, mock_services, regression100_data):
        """Test _generate_plot_data method with regression100.csv."""
        evaluator = regression_plots.PlotPredVsObs(
            "brisk_plot_pred_vs_obs",
            "Plot predicted vs. observed values."
        )
        evaluator.services = mock_services
        
        _, y, predictions = regression100_data
        
        plot_data, max_range = evaluator._generate_plot_data(predictions, y)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert max_range == 359.26584339867804
        assert plot_data.shape == (80, 2)


class TestPlotResiduals:
    """Test PlotResiduals evaluator."""
    def test_generate_plot_data_regression(self, mock_services, regression_data):
        """Test _generate_plot_data method with regression.csv."""
        evaluator = regression_plots.PlotResiduals(
            "brisk_plot_residuals",
            "Plot residuals of model predictions."
        )
        evaluator.services = mock_services
        
        _, y, predictions = regression_data
        
        plot_data = evaluator._generate_plot_data(predictions, y)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (4, 2)
        for i, residual in enumerate(plot_data["Residual (Observed - Predicted)"]):
            assert np.isclose(residual, [0.1, -0.13, 0.07, 0.21][i])

    def test_generate_plot_data_regression100(self, mock_services, regression100_data):
        """Test _generate_plot_data method with regression100.csv."""
        evaluator = regression_plots.PlotResiduals(
            "brisk_plot_residuals",
            "Plot residuals of model predictions."
        )
        evaluator.services = mock_services
        
        _, y, predictions = regression100_data
        
        plot_data = evaluator._generate_plot_data(predictions, y)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (80, 2)
        expected_residuals = [
            -1.96038523e-01,  7.64259578e+01,  3.71296002e-01,  6.94970902e+00,
            2.87476546e+01,  1.09544454e+02, -1.25093780e+02, -4.06277857e+01,
            -7.96502198e+00,  6.43707838e+01, -3.10682598e+02, -5.00226558e+01,
            1.43164354e+01, -9.01584335e+01, -1.29118456e+01, -7.01143856e+00,
            3.45193160e+01, -5.45272021e+00, -6.80420962e+01, -2.71208210e+00,
            1.55701251e+02,  1.37623083e+02,  2.56385059e+01, -8.17784100e+01,
            1.57526929e+02, -1.58639861e+00, -3.63221478e+01,  2.25813163e+02,
            2.80794882e+01, -1.40641978e+01,  4.64026677e+01,  6.52656654e+01,
            -1.51056946e+02,  3.10438791e+01, -7.90968673e+01, -2.42884558e+01,
            4.12579487e+01,  5.06548902e+01,  7.27605331e+01,  6.83516701e-02,
            3.55246259e+01, -5.06051132e+01, -9.35146010e+00,  6.09816270e+01,
            2.99034095e+01, -2.35539017e+01,  2.16301701e+02,  1.12445581e+01,
            6.56262941e+00,  9.62192167e+01, -9.54105576e+00,  8.11632999e+00,
            -5.35505489e+00, -3.04555617e+01, -3.08400141e+01,  8.72650741e+01,
            4.08545364e+01, -1.43098395e+00, -6.49195566e+01,  6.67377423e+01,
            -6.54375676e+01,  1.06070159e+02, -2.88681250e+00,  8.64884148e+01,
            -2.33559567e+01,  5.59820389e-03,  7.77005709e-01, -1.18389987e+02,
            5.62906878e+01, -3.20252107e+01,  9.06406703e+00,  5.76721898e+01,
            3.57645372e+00,  8.18235948e+01,  4.42952363e+01, -1.02224477e+00,
            3.92217066e+01, -1.43506373e-01,  3.09299180e+01, -3.11596774e+01
        ]
        for i, residual in enumerate(plot_data["Residual (Observed - Predicted)"]):
            assert np.isclose(residual, expected_residuals[i])

    def test_generate_plot_data_regression100_group(self, mock_services, regression100_group_data):
        """Test _generate_plot_data method with regression100_group.csv."""
        evaluator = regression_plots.PlotResiduals(
            "brisk_plot_residuals",
            "Plot residuals of model predictions."
        )
        evaluator.services = mock_services
        
        _, y, predictions = regression100_group_data
        
        plot_data = evaluator._generate_plot_data(predictions, y)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (61, 2)
        expected_residuals = [
            -5.41936726e+01, -5.12367186e+01,  1.55514516e+01, -1.06077453e+02,
            6.88495711e+01,  2.30402226e+00, -2.01845381e+02, -2.22524151e+01,
            6.03748732e+01,  3.46826735e+01,  7.70345950e+01, -8.73165857e+01,
            -5.40340470e+01, -1.42560686e+02,  3.26429753e+01,  1.10101066e+01,
            1.10702619e+02, -2.02345530e-01,  8.44802369e+01,  6.83391553e+00,
            -1.01294538e+02,  1.17490527e+02, -1.94137908e+01, -9.35865039e+01,
            -3.92033355e-01,  1.96182452e+02, -2.70714475e+00, -8.35986010e+01,
            7.79666541e+01, -1.18667095e+00,  1.08620122e+02,  9.14219836e+01,
            2.35677765e+02,  5.38410047e+00,  6.28955295e+01, -1.74597459e+02,
            2.66069817e+02,  1.55998881e+02, -3.58133605e+01,  4.18007406e+01,
            -1.44760944e+02,  8.76445688e+01, -6.51928001e+01,  7.65341519e+01,
            -1.68386030e+02, -1.50105365e+02,  2.91716384e+02,  1.78719143e+00,
            -7.69117951e+01,  9.08581301e+01, -8.17368130e+01, -2.88738266e+02,
            3.41914765e+02, -1.11599889e+02,  9.70029381e+00, -2.06367441e+01,
            3.01119663e+02,  2.91921228e+00, -2.13664029e+01,  3.86356202e+01,
            -5.97499654e+01
        ]
        for i, residual in enumerate(plot_data["Residual (Observed - Predicted)"]):
            assert np.isclose(residual, expected_residuals[i])


class TestPlotModelComparison:
    """Test PlotModelComparison evaluator."""
    def test_generate_plot_data_regression(self, metric_config, mock_services, fitted_models, regression_data):
        """Test _generate_plot_data method with regression.csv."""
        evaluator = common_plots.PlotModelComparison(
            "brisk_plot_model_comparison",
            "Compare model performance across algorithms."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, _ = regression_data
        models = [fitted_models["linear"], fitted_models["ridge"], fitted_models["elasticnet"]]
        metric = "mean_absolute_error"
        
        plot_data = evaluator._generate_plot_data(*models, X=X, y=y, metric=metric)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (3, 2)
        for i, score in enumerate(plot_data["Score"]):
            assert np.isclose(score, [0.236, 0.236, 0.239][i])

    def test_generate_plot_data_regression100(self, metric_config, mock_services, fitted_models, regression100_data):
        """Test _generate_plot_data method with regression100.csv."""
        evaluator = common_plots.PlotModelComparison(
            "brisk_plot_model_comparison",
            "Compare model performance across algorithms."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, _ = regression100_data
        
        # Refit models on larger dataset
        fitted_models["ridge"].fit(X, y)
        fitted_models["elasticnet"].fit(X, y)
        fitted_models["linear"].fit(X, y)
        
        models = [fitted_models["linear"], fitted_models["ridge"], fitted_models["elasticnet"]]
        metric = "mean_absolute_error"
        
        plot_data = evaluator._generate_plot_data(*models, X=X, y=y, metric=metric)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (3, 2)
        for i, score in enumerate(plot_data["Score"]):
            assert np.isclose(score, [0.072, 1.661, 6.449][i])

    def test_generate_plot_data_regression100_group(self, metric_config, mock_services, fitted_models, regression100_group_data):
        """Test _generate_plot_data method with regression100_group.csv."""
        evaluator = common_plots.PlotModelComparison(
            "brisk_plot_model_comparison",
            "Compare model performance across algorithms."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, _ = regression100_group_data
        
        # Refit models on grouped dataset
        fitted_models["ridge"].fit(X, y)
        fitted_models["elasticnet"].fit(X, y)
        fitted_models["linear"].fit(X, y)
        
        models = [fitted_models["linear"], fitted_models["ridge"], fitted_models["elasticnet"]]
        metric = "mean_absolute_error"
        
        plot_data = evaluator._generate_plot_data(*models, X=X, y=y, metric=metric)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (3, 2)
        for i, score in enumerate(plot_data["Score"]):
            assert np.isclose(score, [3.656, 3.818, 5.253][i])


class TestConfusionMatrix:
    """Test ConfusionMatrix evaluator."""
    def test_calculate_measures_classification(self, mock_services, classification_data):
        """Test _calculate_measures method with classification.csv."""
        evaluator = classification_measures.ConfusionMatrix(
            "brisk_confusion_matrix",
            "Compute confusion matrix."
        )
        evaluator.services = mock_services
        
        _, y, predictions = classification_data
        
        data = evaluator._calculate_measures(predictions, y)
        
        assert isinstance(data, dict)
        assert data["confusion_matrix"] == [[2, 1], [0, 1]]
        assert data["labels"] == ["A", "B"]

    def test_calculate_measures_classification100(self, mock_services, classification100_data):
        """Test _calculate_measures method with classification100.csv."""
        evaluator = classification_measures.ConfusionMatrix(
            "brisk_confusion_matrix",
            "Compute confusion matrix."
        )
        evaluator.services = mock_services
        
        _, y, predictions = classification100_data
        
        data = evaluator._calculate_measures(predictions, y)
        
        assert isinstance(data, dict)
        assert data["confusion_matrix"] == [[19, 21], [19, 21]]
        assert data["labels"] == [0, 1]

    def test_calculate_measures_classification100_group(self, mock_services, classification100_group_data):
        """Test _calculate_measures method with classification100_group.csv."""
        evaluator = classification_measures.ConfusionMatrix(
            "brisk_confusion_matrix",
            "Compute confusion matrix."
        )
        evaluator.services = mock_services
        
        _, y, predictions = classification100_group_data
        
        data = evaluator._calculate_measures(predictions, y)
        
        assert isinstance(data, dict)
        assert data["confusion_matrix"] == [[13, 14], [9, 15]]
        assert data["labels"] == [0, 1]


class TestPlotConfusionHeatmap:
    """Test PlotConfusionHeatmap evaluator."""
    def test_generate_plot_data_classification(self, mock_services, classification_data):
        """Test _generate_plot_data method with classification.csv."""
        evaluator = classification_plots.PlotConfusionHeatmap(
            "brisk_plot_confusion_heatmap",
            "Plot confusion heatmap."
        )
        evaluator.services = mock_services
        
        _, y, predictions = classification_data
        
        data = evaluator._generate_plot_data(predictions, y)
        
        assert isinstance(data, pd.DataFrame)
        assert data.shape == (4, 4)
        assert data["True Label"].tolist() == ["A", "A", "B", "B"]
        assert data["Predicted Label"].tolist() == ["A", "B", "A", "B"]
        assert data["Percentage"].tolist() == [50.0, 25.0, 0.0, 25.0]
        assert data["Label"].tolist() == ["2\n(50.0%)", "1\n(25.0%)", "0\n(0.0%)", "1\n(25.0%)"]

    def test_generate_plot_data_classification100(self, mock_services, classification100_data):
        """Test _generate_plot_data method with classification100.csv."""
        evaluator = classification_plots.PlotConfusionHeatmap(
            "brisk_plot_confusion_heatmap",
            "Plot confusion heatmap."
        )
        evaluator.services = mock_services
        
        _, y, predictions = classification100_data
        
        data = evaluator._generate_plot_data(predictions, y)
        
        assert isinstance(data, pd.DataFrame)
        assert data.shape == (4, 4)
        assert data["True Label"].tolist() == [0, 0, 1, 1]
        assert data["Predicted Label"].tolist() == [0, 1, 0, 1]
        assert data["Percentage"].tolist() == [23.75, 26.25, 23.75, 26.25]
        assert data["Label"].tolist() == ["19\n(23.8%)", "21\n(26.2%)", "19\n(23.8%)", "21\n(26.2%)"]

    def test_generate_plot_data_classification100_group(self, mock_services, classification100_group_data):
        """Test _generate_plot_data method with classification100_group.csv."""
        evaluator = classification_plots.PlotConfusionHeatmap(
            "brisk_plot_confusion_heatmap",
            "Plot confusion heatmap."
        )
        evaluator.services = mock_services
        
        _, y, predictions = classification100_group_data
        
        data = evaluator._generate_plot_data(predictions, y)
        
        assert isinstance(data, pd.DataFrame)
        assert data.shape == (4, 4)
        assert data["True Label"].tolist() == [0, 0, 1, 1]
        assert data["Predicted Label"].tolist() == [0, 1, 0, 1]
        for i, percentage in enumerate(data["Percentage"]):
            assert np.isclose(percentage, [25.49019608, 27.45098039, 17.64705882, 29.41176471][i])
        assert data["Label"].tolist() == ["13\n(25.5%)", "14\n(27.5%)", "9\n(17.6%)", "15\n(29.4%)"]


class TestPlotRocCurve:
    """Test PlotRocCurve evaluator."""  
    def test_generate_plot_data_classification(self, mock_services, classification_data):
        """Test _generate_plot_data method with classification.csv."""
        evaluator = classification_plots.PlotRocCurve(
            "brisk_plot_roc_curve",
            "Plot ROC curve."
        )
        evaluator.services = mock_services
        
        _, y, _ = classification_data
        
        # Create a mock model that returns the y_score we want to test
        mock_model = mock.MagicMock()
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
            [0.8, 0.2],
        ])
        
        X = pd.DataFrame(np.random.randn(4, 2))
        
        plot_data, auc_data, auc = evaluator._generate_plot_data(mock_model, X, y, pos_label="A")
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (6, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["False Positive Rate"].tolist()[i], [0.0, 0.0, 0.0, 1.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["True Positive Rate"].tolist()[i], [0.0, 0.33333333, 0.66666667, 1.0, 0.0, 1.0][i])
            assert plot_data["Type"].tolist()[i] == ["ROC Curve", "ROC Curve", "ROC Curve", "ROC Curve", "Random Guessing", "Random Guessing"][i]
        assert np.isclose(auc, 0.1666666666)

    def test_generate_plot_data_classification100(self, mock_services, classification100_data):
        """Test _generate_plot_data method with classification100.csv."""
        evaluator = classification_plots.PlotRocCurve(
            "brisk_plot_roc_curve",
            "Plot ROC curve."
        )
        evaluator.services = mock_services
        
        _, y, _ = classification100_data
        
        # Create mock model that returns repeated pattern for 80 samples
        y_score_pattern = np.array([
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
            [0.8, 0.2],
        ] * 8)
        
        mock_model = mock.MagicMock()
        mock_model.predict_proba.return_value = y_score_pattern
        
        X = pd.DataFrame(np.random.randn(80, 5))
        
        plot_data, auc_data, auc = evaluator._generate_plot_data(mock_model, X, y, pos_label=0)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (6, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["False Positive Rate"].tolist()[i], [0.0, 0.175, 0.550, 1.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["True Positive Rate"].tolist()[i], [0.0, 0.425, 0.650, 1.0, 0.0, 1.0][i])
            assert plot_data["Type"].tolist()[i] == ["ROC Curve", "ROC Curve", "ROC Curve", "ROC Curve", "Random Guessing", "Random Guessing"][i]
        assert np.isclose(auc, 0.39)

    def test_generate_plot_data_classification100_group(self, mock_services, classification100_group_data):
        """Test _generate_plot_data method with classification100_group.csv."""
        evaluator = classification_plots.PlotRocCurve(
            "brisk_plot_roc_curve",
            "Plot ROC curve."
        )
        evaluator.services = mock_services
        
        _, y, _ = classification100_group_data
        
        y_score_pattern = np.array([
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
        ] * 17)
        
        mock_model = mock.MagicMock()
        mock_model.predict_proba.return_value = y_score_pattern
        
        X = pd.DataFrame(np.random.randn(51, 5))
        
        plot_data, auc_data, auc = evaluator._generate_plot_data(mock_model, X, y, pos_label=0)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (5, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["False Positive Rate"].tolist()[i], [0.0, 0.333333, 1.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["True Positive Rate"].tolist()[i], [0.0, 0.333333, 1.0, 0.0, 1.0][i])
            assert plot_data["Type"].tolist()[i] == ["ROC Curve", "ROC Curve", "ROC Curve", "Random Guessing", "Random Guessing"][i]
        assert np.isclose(auc, 0.5)


class TestPlotPrecisionRecallCurve:
    """Test PlotPrecisionRecallCurve evaluator.""" 
    def test_generate_plot_data_classification(self, mock_services, classification_data):
        """Test _generate_plot_data method with classification.csv."""
        evaluator = classification_plots.PlotPrecisionRecallCurve(
            "brisk_plot_precision_recall_curve",
            "Plot precision-recall curve."
        )
        evaluator.services = mock_services
        
        _, y, _ = classification_data
        
        mock_model = mock.MagicMock()
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
            [0.8, 0.2],
        ])
        
        X = pd.DataFrame(np.random.randn(4, 2))
        
        plot_data, ap_score = evaluator._generate_plot_data(mock_model, X, y, pos_label="A")
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (6, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["Recall"].tolist()[i], [1.0, 0.666667, 0.333333, 0.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["Precision"].tolist()[i], [0.75, 1.0, 1.0, 1.0, 0.916667, 0.916667][i])
            assert plot_data["Type"].tolist()[i] == ["PR Curve", "PR Curve", "PR Curve", "PR Curve", "AP Score = 0.92", "AP Score = 0.92"][i]
        assert np.isclose(ap_score, 0.9166666666)

    def test_generate_plot_data_classification100(self, mock_services, classification100_data):
        """Test _generate_plot_data method with classification100.csv."""
        evaluator = classification_plots.PlotPrecisionRecallCurve(
            "brisk_plot_precision_recall_curve",
            "Plot precision-recall curve."
        )
        evaluator.services = mock_services
        
        _, y, _ = classification100_data
        
        y_score_pattern = np.array([
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
            [0.8, 0.2],
        ] * 8)
        
        mock_model = mock.MagicMock()
        mock_model.predict_proba.return_value = y_score_pattern
        
        X = pd.DataFrame(np.random.randn(80, 5))
        
        plot_data, ap_score = evaluator._generate_plot_data(mock_model, X, y, pos_label=0)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (6, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["Recall"].tolist()[i], [1.0, 0.650, 0.425, 0.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["Precision"].tolist()[i], [0.5, 0.541667, 0.708333, 1.0, 0.597917, 0.597917][i])
            assert plot_data["Type"].tolist()[i] == ["PR Curve", "PR Curve", "PR Curve", "PR Curve", "AP Score = 0.60", "AP Score = 0.60"][i]
        assert np.isclose(ap_score, 0.5979166666666667)

    def test_generate_plot_data_classification100_group(self, mock_services, classification100_group_data):
        """Test _generate_plot_data method with classification100_group.csv."""
        evaluator = classification_plots.PlotPrecisionRecallCurve(
            "brisk_plot_precision_recall_curve",
            "Plot precision-recall curve."
        )
        evaluator.services = mock_services
        
        _, y, _ = classification100_group_data
        
        y_score_pattern = np.array([
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
        ] * 17)
        
        mock_model = mock.MagicMock()
        mock_model.predict_proba.return_value = y_score_pattern
        
        X = pd.DataFrame(np.random.randn(51, 5))
        
        plot_data, ap_score = evaluator._generate_plot_data(mock_model, X, y, pos_label=0)
        
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (6, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["Recall"].tolist()[i], [1.0, 0.666667, 0.333333, 0.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["Precision"].tolist()[i], [0.529412, 0.529412, 0.529412, 1.0, 0.529412, 0.529412][i])
            assert plot_data["Type"].tolist()[i] == ["PR Curve", "PR Curve", "PR Curve", "PR Curve", "AP Score = 0.53", "AP Score = 0.53"][i]
        assert np.isclose(ap_score, 0.5294117647058822)


class TestPlotFeatureImportance:
    """Test PlotFeatureImportance evaluator."""
    def test_generate_plot_data_regression100(self, metric_config, mock_services, fitted_models, regression100_data):
        """Test _generate_plot_data method with regression100.csv."""
        evaluator = common_plots.PlotFeatureImportance(
            "brisk_plot_feature_importance",
            "Plot feature importance."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, _ = regression100_data
        model = fitted_models["ridge"]
        model.fit(X, y)
        
        feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        
        importance_data, plot_width, plot_height = evaluator._generate_plot_data(
            model, X, y, threshold=3, feature_names=feature_names,
            metric="mean_absolute_error", num_rep=1
        )
        
        assert isinstance(importance_data, pd.DataFrame)
        assert importance_data.shape == (3, 2)
        assert plot_width == 8
        assert plot_height == 6
        
        importance_data, plot_width, plot_height = evaluator._generate_plot_data(
            model, X, y, threshold=5, feature_names=feature_names,
            metric="mean_absolute_error", num_rep=1
        )
        
        assert isinstance(importance_data, pd.DataFrame)
        assert importance_data.shape == (5, 2)
        assert plot_width == 8
        assert plot_height == 6

    def test_generate_plot_data_regression100_group(self, metric_config, mock_services, fitted_models, regression100_group_data):
        """Test _generate_plot_data method with regression100_group.csv."""
        evaluator = common_plots.PlotFeatureImportance(
            "brisk_plot_feature_importance",
            "Plot feature importance."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, _ = regression100_group_data
        model = fitted_models["ridge"]
        model.fit(X, y)
        
        feature_names = ["cat_feature_1","cont_feature_1", "cont_feature_2", "cont_feature_3", "cont_feature_4"]
        
        importance_data, plot_width, plot_height = evaluator._generate_plot_data(
            model, X, y, threshold=0.1, feature_names=feature_names,
            metric="mean_absolute_error", num_rep=1
        )
        
        assert isinstance(importance_data, pd.DataFrame)
        assert importance_data.shape == (1, 2)
        assert plot_width == 8
        assert plot_height == 6
        
        importance_data, plot_width, plot_height = evaluator._generate_plot_data(
            model, X, y, threshold=0.8, feature_names=feature_names,
            metric="mean_absolute_error", num_rep=1
        )
        
        assert isinstance(importance_data, pd.DataFrame)
        assert importance_data.shape == (4, 2)
        assert plot_width == 8
        assert plot_height == 6


class TestHyperparameterTuning:
    """Test HyperparameterTuning evaluator."""
    @pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
    def test_calculate_measures(self, metric_config, mock_services, fitted_models, regression_data, algorithm_config):
        """Test _calculate_measures method."""
        evaluator = tools.HyperparameterTuning(
            "brisk_hyperparameter_tuning",
            "Hyperparameter tuning."
        )
        evaluator.metric_config = metric_config
        evaluator.services = mock_services
        
        X, y, _ = regression_data
        model = fitted_models["rf"]
        
        with patch.object(evaluator, '_plot_hyperparameter_performance'):
            tuned_model = evaluator._calculate_measures(
                model=model, 
                method="grid", 
                X_train=X, 
                y_train=y,
                scorer="mean_squared_error", 
                kf=3, 
                num_rep=1, 
                n_jobs=1,
                param_grid=algorithm_config["rf"].hyperparam_grid
            )
        
        assert hasattr(tuned_model, 'fit')
        assert hasattr(tuned_model, 'predict')


class TestPlotShapleyValues:
    """Test PlotShapleyValues evaluator."""
    

    def test_plot_multiple_types(self, mock_services, fitted_models, regression_data):
        """Test plot method with multiple plot types."""
        evaluator = common_plots.PlotShapleyValues(
            "brisk_plot_shapley_values",
            "Plot SHAP values for feature importance."
        )
        evaluator.services = mock_services
        
        X, y, _ = regression_data
        X.attrs = {"is_test": True}  # Add required attrs
        model = fitted_models["linear"]
        
        # Mock all the methods that plot() calls
        with patch.object(evaluator, '_generate_plot_data') as mock_generate, \
             patch.object(evaluator, '_create_plot') as mock_create, \
             patch.object(evaluator, '_generate_metadata') as mock_metadata, \
             patch.object(evaluator, '_save_plot') as mock_save, \
             patch.object(evaluator, '_log_results') as mock_log:
            
            # Mock return values
            mock_generate.return_value = {'mock': 'data'}
            mock_create.return_value = mock.MagicMock()  # Mock plot object
            mock_metadata.return_value = {'test': True}
            
            # Test multiple plot types
            evaluator.plot(model, X, y, filename="test_shap", plot_type="bar,waterfall")
            
            # Verify _generate_plot_data called once
            mock_generate.assert_called_once_with(model, X, y, "bar,waterfall")
            
            # Verify _create_plot called twice (once for each plot type)
            assert mock_create.call_count == 2
            mock_create.assert_any_call({'mock': 'data'}, 'bar')
            mock_create.assert_any_call({'mock': 'data'}, 'waterfall')
            
            # Verify _save_plot called twice with different filenames
            assert mock_save.call_count == 2
            mock_save.assert_any_call("test_shap_bar", {'test': True}, plot=mock_create.return_value)
            mock_save.assert_any_call("test_shap_waterfall", {'test': True}, plot=mock_create.return_value)
            
            # Verify _log_results called twice
            assert mock_log.call_count == 2
            mock_log.assert_any_call("SHAP Values", "test_shap_bar")
            mock_log.assert_any_call("SHAP Values", "test_shap_waterfall")
    
    def test_plot_single_type(self, mock_services, fitted_models, regression_data):
        """Test plot method with single plot type (no filename suffix)."""
        evaluator = common_plots.PlotShapleyValues(
            "brisk_plot_shapley_values",
            "Plot SHAP values for feature importance."
        )
        evaluator.services = mock_services
        
        X, y, _ = regression_data
        X.attrs = {"is_test": True}  # Add required attrs
        model = fitted_models["linear"]
        
        # Mock all the methods that plot() calls
        with patch.object(evaluator, '_generate_plot_data') as mock_generate, \
             patch.object(evaluator, '_create_plot') as mock_create, \
             patch.object(evaluator, '_generate_metadata') as mock_metadata, \
             patch.object(evaluator, '_save_plot') as mock_save, \
             patch.object(evaluator, '_log_results') as mock_log:
            
            # Mock return values
            mock_generate.return_value = {'mock': 'data'}
            mock_create.return_value = mock.MagicMock()  # Mock plot object
            mock_metadata.return_value = {'test': True}
            
            # Test single plot type
            evaluator.plot(model, X, y, filename="test_shap", plot_type="bar")
            
            # Verify methods called once each
            mock_generate.assert_called_once_with(model, X, y, "bar")
            mock_create.assert_called_once_with({'mock': 'data'}, 'bar')
            
            # Verify filename has no suffix for single plot
            mock_save.assert_called_once_with("test_shap", {'test': True}, plot=mock_create.return_value)
            mock_log.assert_called_once_with("SHAP Values", "test_shap")
