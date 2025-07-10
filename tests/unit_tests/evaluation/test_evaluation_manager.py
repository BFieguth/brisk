import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from importlib import util

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection

from brisk.evaluation.evaluation_manager import EvaluationManager
from brisk.configuration.algorithm_wrapper import AlgorithmCollection
from brisk.evaluation.metric_manager import MetricManager
from brisk.data.data_manager import DataManager

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
def algorithm_config(mock_brisk_project, tmp_path):
    algorithm_file = tmp_path / "algorithms.py"
    spec = util.spec_from_file_location("algorithms", algorithm_file)
    algorithm_module = util.module_from_spec(spec)
    spec.loader.exec_module(algorithm_module)
    return algorithm_module.ALGORITHM_CONFIG


@pytest.fixture
def regression_data(data_manager, tmp_path):
    data_file = tmp_path / "datasets" / "regression.csv"
    split = data_manager.split(
        data_file,
        categorical_features=None,
        table_name=None,
        group_name=None,
        filename=None
    )
    X_train, y_train = split.get_train()
    X_train.attrs["is_test"] = False
    y_train.attrs["is_test"] = False
    # Fixed predictions for testing
    predictions = [0.7, 0.23, 0.43, 0.79]
    return X_train, y_train, predictions


@pytest.fixture
def regression100_data(data_manager, tmp_path):
    data_file = tmp_path / "datasets" / "regression100.csv"
    split = data_manager.split(
        data_file,
        categorical_features=None,
    )
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
    split = data_manager_group.split(
        data_file,
        categorical_features=["cat_feature_1"],
    )
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
    split = data_manager.split(
        data_file,
        categorical_features=None,
        table_name=None,
        group_name=None,
        filename=None
    )
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
    split = data_manager.split(
        data_file,
        categorical_features=["cat_feature_1", "cat_feature_2"],
    )
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
    split = data_manager_group.split(
        data_file,
        categorical_features=["cat_feature_1", "cat_feature_2"],
    )
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
def eval_manager(tmp_path, tmpdir, algorithm_config):
    metric_file = tmp_path / "metric.py"
    spec = util.spec_from_file_location("metric", metric_file)
    metric_module = util.module_from_spec(spec)
    spec.loader.exec_module(metric_module)
    metric_config = metric_module.METRIC_CONFIG
    split_metadata = {
        "num_features": 2,
        "num_samples": 4
    }
    return EvaluationManager(
        algorithm_config, metric_config, output_dir=str(tmpdir), 
        split_metadata=split_metadata, logger=MagicMock(),
        group_index_train=None, group_index_test=None
    )


@pytest.fixture
def rf_regressor(algorithm_config, regression_data):
    X, y, _= regression_data
    model = algorithm_config["rf"].instantiate()
    model.fit(X, y)
    return model


@pytest.fixture
def rf_classifier(algorithm_config, classification_data):
    X, y, _ = classification_data
    model = algorithm_config["rf_classifier"].instantiate()
    model.fit(X, y)
    return model


@pytest.fixture()
def ridge(algorithm_config, regression_data):
    X, y, _ = regression_data
    model = algorithm_config["ridge"].instantiate()
    model.fit(X, y)
    return model


@pytest.fixture
def linear(algorithm_config, regression_data):
    X, y, _ = regression_data
    model = algorithm_config["linear"].instantiate()
    model.fit(X, y)
    return model


@pytest.fixture
def elasticnet(algorithm_config, regression_data):
    X, y, _ = regression_data
    model = algorithm_config["elasticnet"].instantiate()
    model.fit(X, y)
    return model


class TestEvaluationManager:
    def test_init(self, eval_manager):
        assert isinstance(eval_manager, EvaluationManager)
        assert isinstance(eval_manager.algorithm_config, AlgorithmCollection)
        assert isinstance(eval_manager.metric_config, MetricManager)
        # Check colours are defined for plots
        assert hasattr(eval_manager, "primary_color")
        assert hasattr(eval_manager, "secondary_color")
        assert hasattr(eval_manager, "background_color")
        assert hasattr(eval_manager, "accent_color")
        assert hasattr(eval_manager, "important_color")

    def test_evaluate_model(self, eval_manager, rf_regressor, regression_data, tmpdir):
        X, y, _ = regression_data
        filename = tmpdir.join("evaluation_result")
        metrics = ["mean_absolute_error"]

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_to_json") as mock_save_json:

            eval_manager.evaluate_model(rf_regressor, X, y, metrics, filename)

            mock_save_json.assert_called_once()
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            eval_manager.logger.info.assert_called()

    def test_calc_evaluate_model(self, eval_manager, regression_data, regression100_data, regression100_group_data):
        # regression.csv
        _, y, predictions = regression_data
        metrics = ["mean_absolute_error", "MSE", "AdjR2"]
        result = eval_manager._calc_evaluate_model(predictions, y, metrics)
        assert isinstance(result, dict)
        assert np.isclose(result["Mean Absolute Error"], 0.12750)
        assert np.isclose(result["Mean Squared Error"], 0.0189750)
        assert np.isclose(result["Adjusted R2 Score"], 0.5049999999999999)
        
        # regression100.csv
        _, y, predictions = regression100_data
        metrics = ["mean_absolute_error", "MAPE", "r2_score"]
        result = eval_manager._calc_evaluate_model(predictions, y, metrics)
        assert isinstance(result, dict)
        assert np.isclose(result["Mean Absolute Error"], 51.98190036274825)
        assert np.isclose(result["Mean Absolute Percentage Error"], 0.4720773856131825)
        assert np.isclose(result["R2 Score"], 0.696868586896576)

        # regression100_group.csv
        X, y, predictions = regression100_group_data
        assert X.shape == (61, 5)
        metrics = ["CCC", "MAE"]
        result = eval_manager._calc_evaluate_model(predictions, y, metrics)
        assert isinstance(result, dict)
        assert np.isclose(result["Concordance Correlation Coefficient"], 0.15951165783251642)
        assert np.isclose(result["Mean Absolute Error"], 90.62786907070439)

    def test_evaluate_model_cv(self, eval_manager, rf_regressor, regression_data, tmpdir):
        X, y, _ = regression_data
        filename = tmpdir.join("cv_evaluation_result")
        metrics = ["mean_absolute_error"]

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_to_json") as mock_save_json:
            
            eval_manager.evaluate_model_cv(rf_regressor, X, y, metrics, filename, cv=2)
           
            mock_save_json.assert_called_once()
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            eval_manager.logger.info.assert_called()

    def test_calc_evaluate_model_cv(self, eval_manager, ridge, regression_data, regression100_data, regression100_group_data):
        # regression.csv
        X, y, _ = regression_data
        metrics = ["mean_absolute_error", "MSE"]
        result = eval_manager._calc_evaluate_model_cv(ridge, X, y, metrics, cv=2)
        assert isinstance(result, dict)
        assert np.isclose(result["Mean Absolute Error"]["mean_score"], 0.6225)
        assert np.isclose(result["Mean Absolute Error"]["std_dev"], 0.09750000000000014)
        for i, score in enumerate(result["Mean Absolute Error"]["all_scores"]):
            assert np.isclose(score, [0.525, 0.72][i])
        assert np.isclose(result["Mean Squared Error"]["mean_score"], 0.4314625000000001)
        assert np.isclose(result["Mean Squared Error"]["std_dev"], 0.11583750000000012)
        for i, score in enumerate(result["Mean Squared Error"]["all_scores"]):
            assert np.isclose(score, [0.315625, 0.5473][i])

        # regression100.csv
        X, y, _ = regression100_data
        metrics = ["R2", "CCC"]
        result = eval_manager._calc_evaluate_model_cv(ridge, X, y, metrics, cv=4)
        assert isinstance(result, dict)
        assert np.isclose(result["R2 Score"]["mean_score"], 0.9995030407082244)
        assert np.isclose(result["R2 Score"]["std_dev"], 0.0001309841689887547)
        for i, score in enumerate(result["R2 Score"]["all_scores"]):
            assert np.isclose(score, [0.99945893, 0.99956915, 0.99931588, 0.99966821][i])
        assert np.isclose(result["Concordance Correlation Coefficient"]["mean_score"], 0.9997462832551977)
        assert np.isclose(result["Concordance Correlation Coefficient"]["std_dev"], 6.744853108793018e-05)
        for i, score in enumerate(result["Concordance Correlation Coefficient"]["all_scores"]):
            assert np.isclose(score, [0.9997237, 0.99978056, 0.99964976, 0.99983111][i])

        # regression100_group.csv
        X, y, _ = regression100_group_data
        metrics = ["MAE", "MAPE"]
        result = eval_manager._calc_evaluate_model_cv(ridge, X, y, metrics, cv=4)
        assert isinstance(result, dict)
        assert np.isclose(result["Mean Absolute Error"]["mean_score"], 4.859018024947762)
        assert np.isclose(result["Mean Absolute Error"]["std_dev"], 1.234307892954546)
        for i, score in enumerate(result["Mean Absolute Error"]["all_scores"]):
            assert np.isclose(score, [3.49985899, 4.3503448, 6.85208558, 4.73378273][i])
    
        assert np.isclose(result["Mean Absolute Percentage Error"]["mean_score"], 0.6175659655526924)
        assert np.isclose(result["Mean Absolute Percentage Error"]["std_dev"], 0.8571714086912385)
        for i, score in enumerate(result["Mean Absolute Percentage Error"]["all_scores"]):
            assert np.isclose(score, [0.18203515, 2.10087968, 0.08530377, 0.10204527][i])

    def test_compare_models(self, eval_manager, rf_regressor, ridge, regression_data, tmpdir):
        X, y, _ = regression_data
        filename = tmpdir.join("comparison_result")
        metrics = ["mean_absolute_error"]

        with patch("os.makedirs") as mock_makedirs, \
            patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_to_json") as mock_save_json:
        
            eval_manager.compare_models(rf_regressor, ridge, X=X, y=y, metrics=metrics, filename=filename)
           
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_json.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_calc_compare_models(self, eval_manager, linear, ridge, elasticnet, regression_data, regression100_data):
        # regression.csv
        X, y, _ = regression_data
        metrics = ["mean_absolute_error", "MSE"]
        result = eval_manager._calc_compare_models(
            linear, ridge, elasticnet, X=X, y=y, metrics=metrics, 
            calculate_diff=True
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

        # regression100.csv
        X, y, _ = regression100_data
        ridge.fit(X, y)
        linear.fit(X, y)
        metrics = ["mean_absolute_error", "MSE"]
        result = eval_manager._calc_compare_models(
            linear, ridge, X=X, y=y, metrics=metrics, 
            calculate_diff=True
        )
        assert isinstance(result, dict)
        assert np.isclose(result["Linear Regression"]["Mean Absolute Error"], 0.07161930571365704)
        assert np.isclose(result["Ridge Regression"]["Mean Absolute Error"], 1.6613532940107134)

        assert np.isclose(result["Linear Regression"]["Mean Squared Error"], 0.008157956114528275)
        assert np.isclose(result["Ridge Regression"]["Mean Squared Error"], 4.036739214414959)

        assert np.isclose(result["differences"]["Mean Absolute Error"]["Ridge Regression - Linear Regression"], 1.5897339882970565)
        assert np.isclose(result["differences"]["Mean Squared Error"]["Ridge Regression - Linear Regression"], 4.028581258300431)

    def test_plot_pred_vs_obs(self, eval_manager, rf_regressor, regression_data, tmpdir):
        X, y, _ = regression_data
        filename = tmpdir.join("pred_vs_obs_plot")

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:
           
            eval_manager.plot_pred_vs_obs(rf_regressor, X, y, filename)
           
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_calc_plot_pred_vs_obs(self, eval_manager, regression_data, regression100_data):
        # regression.csv
        _, y, prediction = regression_data
        plot_data, max_range = eval_manager._calc_plot_pred_vs_obs(prediction, y)
        assert isinstance(plot_data, pd.DataFrame)
        assert max_range == 1.0
        assert plot_data.shape == (4, 2)

        # regression100.csv
        _, y, prediction = regression100_data
        plot_data, max_range = eval_manager._calc_plot_pred_vs_obs(prediction, y)
        assert isinstance(plot_data, pd.DataFrame)
        assert max_range == 359.26584339867804
        assert plot_data.shape == (80, 2)

    @pytest.mark.filterwarnings("ignore:Removed duplicate entries from 'train_sizes':RuntimeWarning")
    def test_plot_learning_curve(self, eval_manager, rf_regressor, regression_data, tmpdir):
        X, y, _ = regression_data
        filename = tmpdir.join("learning_curve")

        with patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:
            
            eval_manager.plot_learning_curve(
                rf_regressor, X, y, filename=filename, n_jobs=1, cv=2
                )
            
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_plot_feature_importance(self, eval_manager, rf_regressor, regression_data, tmpdir):
        X, y, _ = regression_data
        feature_names = [f'feature{i}' for i in range(X.shape[1])]
        filename = tmpdir.join("feature_importance")

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:
            
            eval_manager.plot_feature_importance(
                rf_regressor, X, y, threshold=3, feature_names=feature_names,
                filename=filename, metric="mean_absolute_error", num_rep=1
            )
            
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_calc_plot_feature_importance(self, eval_manager, ridge, regression100_data, regression100_group_data):
        # regression100.csv
        X, y, _ = regression100_data
        feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        importance_data, plot_width, plot_height = eval_manager._calc_plot_feature_importance(
            ridge, X, y, threshold=3, feature_names=feature_names,
            metric="mean_absolute_error", num_rep=1
        )
        assert isinstance(importance_data, pd.DataFrame)
        assert importance_data.shape == (3, 2)
        assert plot_width == 8
        assert plot_height == 6

        importance_data, plot_width, plot_height = eval_manager._calc_plot_feature_importance(
            ridge, X, y, threshold=5, feature_names=feature_names,
            metric="mean_absolute_error", num_rep=1
        )
        assert isinstance(importance_data, pd.DataFrame)
        assert importance_data.shape == (5, 2)
        assert plot_width == 8
        assert plot_height == 6

        # regression100_group.csv
        X, y, _ = regression100_group_data
        feature_names = ["cat_feature_1","cont_feature_1", "cont_feature_2", "cont_feature_3", "cont_feature_4"]
        importance_data, plot_width, plot_height = eval_manager._calc_plot_feature_importance(
            ridge, X, y, threshold=0.1, feature_names=feature_names,
            metric="mean_absolute_error", num_rep=1
        )
        assert isinstance(importance_data, pd.DataFrame)
        assert importance_data.shape == (1, 2)
        assert plot_width == 8
        assert plot_height == 6

        importance_data, plot_width, plot_height = eval_manager._calc_plot_feature_importance(
            ridge, X, y, threshold=0.8, feature_names=feature_names,
            metric="mean_absolute_error", num_rep=1
        )
        assert isinstance(importance_data, pd.DataFrame)
        assert importance_data.shape == (4, 2)
        assert plot_width == 8
        assert plot_height == 6

    def test_plot_residuals(self, eval_manager, rf_regressor, regression_data, tmpdir):
        X, y, _ = regression_data
        filename = tmpdir.join("residuals_plot")

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:
            
            eval_manager.plot_residuals(rf_regressor, X, y, filename)
            
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_calc_plot_residuals(self, eval_manager, regression_data, regression100_data, regression100_group_data):
        # regression.csv
        _ , y, predictions = regression_data
        plot_data = eval_manager._calc_plot_residuals(predictions, y)
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (4, 2)
        for i, residual in enumerate(plot_data["Residual (Observed - Predicted)"]):
            assert np.isclose(residual, [0.1, -0.13, 0.07, 0.21][i])

        # regression100.csv
        _ , y, predictions = regression100_data
        plot_data = eval_manager._calc_plot_residuals(predictions, y)
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (80, 2)
        for i, residual in enumerate(plot_data["Residual (Observed - Predicted)"]):
            assert np.isclose(residual, [
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
            ][i])

        # regression100_group.csv
        _ , y, predictions = regression100_group_data
        plot_data = eval_manager._calc_plot_residuals(predictions, y)
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (61, 2)
        for i, residual in enumerate(plot_data["Residual (Observed - Predicted)"]):
            assert np.isclose(residual, [
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
            ][i])

    def test_plot_model_comparison(self, eval_manager, rf_regressor, ridge, regression_data):
        X, y, _ = regression_data
        models = [rf_regressor, ridge]
        metric = "mean_absolute_error"
        filename = "model_comparison"

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager.plot_model_comparison(*models, X=X, y=y, metric=metric, filename=filename)
            
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_calc_plot_model_comparison(self, eval_manager, linear, ridge, elasticnet, regression_data, regression100_data, regression100_group_data):
        models = [linear, ridge, elasticnet]
        metric = "mean_absolute_error"
        # regression.csv
        X, y, _ = regression_data
        plot_data = eval_manager._calc_plot_model_comparison(
            *models, X=X, y=y, metric=metric
        )
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (3, 2)
        for i, score in enumerate(plot_data["Score"]):
            assert np.isclose(score, [0.236, 0.236, 0.239][i])

        # regression100.csv
        X, y, _ = regression100_data
        ridge.fit(X, y)
        elasticnet.fit(X, y)
        linear.fit(X, y)
        plot_data = eval_manager._calc_plot_model_comparison(
            *models, X=X, y=y, metric=metric
        )
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (3, 2)
        for i, score in enumerate(plot_data["Score"]):
            assert np.isclose(score, [0.072, 1.661, 6.449][i])

        # regression100_group.csv
        X, y, _ = regression100_group_data
        ridge.fit(X, y)
        elasticnet.fit(X, y)
        linear.fit(X, y)
        plot_data = eval_manager._calc_plot_model_comparison(
            *models, X=X, y=y, metric=metric
        )
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (3, 2)
        for i, score in enumerate(plot_data["Score"]):
            assert np.isclose(score, [3.656, 3.818, 5.253][i])

    @pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
    def test_hyperparameter_tuning(self, eval_manager, rf_regressor, regression_data):
        X, y, _ = regression_data

        with patch("brisk.evaluation.evaluation_manager.EvaluationManager._plot_hyperparameter_performance") as mock_plot:

            tuned_model = eval_manager.hyperparameter_tuning(
                model=rf_regressor, method="grid", X_train=X, y_train=y,
                scorer="mean_squared_error", kf=3, num_rep=1, n_jobs=1, plot_results=True
            )
           
            mock_plot.assert_called_once()
            assert isinstance(tuned_model, RandomForestRegressor)
            eval_manager.logger.info.assert_called()

    def test_plot_hyperparameter_performance(self, eval_manager):
        param_grid = {"n_estimators": [10, 50, 100]}
        search_result = MagicMock()
        search_result.cv_results_ = {'mean_test_score': [0.8, 0.85, 0.87]}
        algorithm_name = "random_forest"
        metadata = {"test": "data"}
        display_name = "Random Forest"

        with patch("brisk.evaluation.evaluation_manager.EvaluationManager._plot_1d_performance") as mock_plot_1d, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._plot_3d_surface") as mock_plot_3d:

            eval_manager._plot_hyperparameter_performance(param_grid, search_result, algorithm_name, metadata, display_name)
            
            mock_plot_1d.assert_called_once_with(
                param_values=param_grid["n_estimators"],
                mean_test_score=search_result.cv_results_['mean_test_score'],
                param_name="n_estimators",
                algorithm_name=algorithm_name,
                metadata=metadata,
                display_name=display_name
            )
            mock_plot_3d.assert_not_called()

    def test_plot_1d_performance(self, eval_manager):
        param_values = [10, 50, 100]
        mean_test_score = [0.8, 0.85, 0.87]
        param_name = "n_estimators"
        algorithm_name = "random_forest"
        metadata = {"test": "data"}
        display_name = "Random Forest"
        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager._plot_1d_performance(param_values, mean_test_score, param_name, algorithm_name, metadata, display_name)
            
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_plot_3d_surface(self, eval_manager):
        param_grid = {"max_depth": [5, 10, 15], "min_samples_split": [2, 4]}
        search_result = MagicMock()
        search_result.cv_results_ = {'mean_test_score': np.array([0.8, 0.82, 0.85, 0.83, 0.86, 0.87]).reshape(3, 2)}
        param_names = ["max_depth", "min_samples_split"]
        algorithm_name = "random_forest"
        metadata = {"test": "data"}
        display_name = "Random Forest"
        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager._plot_3d_surface(param_grid, search_result, param_names, algorithm_name, metadata, display_name)
            
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_calc_plot_3d_surface(self, eval_manager):
        param_grid = {"max_depth": [5, 10, 15], "min_samples_split": [2, 4]}
        search_result = MagicMock()
        search_result.cv_results_ = {'mean_test_score': np.array([0.8, 0.82, 0.85, 0.83, 0.86, 0.87]).reshape(3, 2)}
        param_names = ["max_depth", "min_samples_split"]
        X, Y, mean_test_score = eval_manager._calc_plot_3d_surface(param_grid, search_result, param_names)
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert isinstance(mean_test_score, np.ndarray)
        assert X.shape == (2, 3)
        np.testing.assert_array_equal(X, np.array([[5, 10, 15], [5, 10, 15]]))
        assert Y.shape == (2, 3)
        np.testing.assert_array_equal(Y, np.array([[2, 2, 2], [4, 4, 4]]))
        assert mean_test_score.shape == (3, 2)
        np.testing.assert_array_equal(mean_test_score, np.array([[0.8, 0.82], [0.85, 0.83], [0.86, 0.87]]))

    def test_confusion_matrix(self, eval_manager, rf_classifier, classification_data):
        X, y, _ = classification_data
        filename = "confusion_matrix"

        with patch("os.makedirs") as mock_makedirs, \
            patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_to_json") as mock_save_json:

            eval_manager.confusion_matrix(rf_classifier, X, y, filename)

            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_json.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_calc_confusion_matrix(self, eval_manager, classification_data, classification100_data, classification100_group_data):
        # classification.csv
        _, y, predictions = classification_data
        data = eval_manager._calc_confusion_matrix(predictions, y)
        assert isinstance(data, dict)
        assert data["confusion_matrix"] == [[2, 1], [0, 1]]
        assert data["labels"] == ["A", "B"]

        # classification100.csv
        _, y, predictions = classification100_data
        data = eval_manager._calc_confusion_matrix(predictions, y)
        assert isinstance(data, dict)
        assert data["confusion_matrix"] == [[19, 21], [19, 21]]
        assert data["labels"] == [0, 1]

        # classification100_group.csv
        _, y, predictions = classification100_group_data
        data = eval_manager._calc_confusion_matrix(predictions, y)
        assert isinstance(data, dict)
        assert data["confusion_matrix"] == [[13, 14], [9, 15]]
        assert data["labels"] == [0, 1]

    def test_plot_confusion_heatmap(self, eval_manager, rf_classifier, classification_data):
        X, y, _ = classification_data
        filename = "confusion_heatmap"

        with patch("os.makedirs") as mock_makedirs, \
            patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager.plot_confusion_heatmap(rf_classifier, X, y, filename)

            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_calc_plot_confusion_heatmap(self, eval_manager, classification_data, classification100_data, classification100_group_data):
        # classification.csv
        _, y, predictions = classification_data
        data = eval_manager._calc_plot_confusion_heatmap(predictions, y)
        assert isinstance(data, pd.DataFrame)
        assert data.shape == (4, 4)
        assert data["True Label"].tolist() == ["A", "A", "B", "B"]
        assert data["Predicted Label"].tolist() == ["A", "B", "A", "B"]
        assert data["Percentage"].tolist() == [50.0, 25.0, 0.0, 25.0]
        assert data["Label"].tolist() == ["2\n(50.0%)", "1\n(25.0%)", "0\n(0.0%)", "1\n(25.0%)"]

        # classification100.csv
        _, y, predictions = classification100_data
        data = eval_manager._calc_plot_confusion_heatmap(predictions, y)
        assert isinstance(data, pd.DataFrame)
        assert data.shape == (4, 4)
        assert data["True Label"].tolist() == [0, 0, 1, 1]
        assert data["Predicted Label"].tolist() == [0, 1, 0, 1]
        assert data["Percentage"].tolist() == [23.75, 26.25, 23.75, 26.25]
        assert data["Label"].tolist() == ["19\n(23.8%)", "21\n(26.2%)", "19\n(23.8%)", "21\n(26.2%)"]

        # classification100_group.csv
        _, y, predictions = classification100_group_data
        data = eval_manager._calc_plot_confusion_heatmap(predictions, y)
        assert isinstance(data, pd.DataFrame)
        assert data.shape == (4, 4)
        assert data["True Label"].tolist() == [0, 0, 1, 1]
        assert data["Predicted Label"].tolist() == [0, 1, 0, 1]
        for i, percentage in enumerate(data["Percentage"]):
            assert np.isclose(percentage, [25.49019608, 27.45098039, 17.64705882, 29.41176471][i])
        assert data["Label"].tolist() == ["13\n(25.5%)", "14\n(27.5%)", "9\n(17.6%)", "15\n(29.4%)"]

    def test_plot_roc_curve(self, eval_manager, rf_classifier, classification_data):
        """Test the plot_roc_curve method of EvaluationManager."""
        X, y, _ = classification_data
        filename = "roc_curve"

        with patch("os.makedirs") as mock_makedirs, \
            patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager.plot_roc_curve(rf_classifier, X, y, filename, pos_label="A")

            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_calc_plot_roc_curve(self, eval_manager, classification_data, classification100_data, classification100_group_data):
        # classification.csv
        _, y, _ = classification_data
        y_score = np.array([
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
            [0.8, 0.2],
        ])[:, 1]
        plot_data, auc_data, auc = eval_manager._calc_plot_roc_curve(y_score, y, pos_label="A")
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (6, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["False Positive Rate"].tolist()[i], [0.0, 0.0, 0.0, 1.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["True Positive Rate"].tolist()[i], [0.0, 0.33333333, 0.66666667, 1.0, 0.0, 1.0][i])
            assert plot_data["Type"].tolist()[i] == ["ROC Curve", "ROC Curve", "ROC Curve", "ROC Curve", "Random Guessing", "Random Guessing"][i]
        assert np.isclose(auc, 0.1666666666)

        # classification100.csv
        _, y, _ = classification100_data
        y_score = np.array([
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
        ] * 8)[:, 1]
        plot_data, auc_data, auc = eval_manager._calc_plot_roc_curve(y_score, y, pos_label=0)
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (6, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["False Positive Rate"].tolist()[i], [0.0, 0.175, 0.550, 1.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["True Positive Rate"].tolist()[i], [0.0, 0.425, 0.650, 1.0, 0.0, 1.0][i])
            assert plot_data["Type"].tolist()[i] == ["ROC Curve", "ROC Curve", "ROC Curve", "ROC Curve", "Random Guessing", "Random Guessing"][i]
        assert np.isclose(auc, 0.39)

        # classification100_group.csv
        _, y, _ = classification100_group_data
        y_score = np.array([
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
        ] * 17)[:, 1]
        plot_data, auc_data, auc = eval_manager._calc_plot_roc_curve(y_score, y, pos_label=0)
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (5, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["False Positive Rate"].tolist()[i], [0.0, 0.333333, 1.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["True Positive Rate"].tolist()[i], [0.0, 0.333333, 1.0, 0.0, 1.0][i])
            assert plot_data["Type"].tolist()[i] == ["ROC Curve", "ROC Curve", "ROC Curve", "Random Guessing", "Random Guessing"][i]
        assert np.isclose(auc, 0.5)

    def test_plot_precision_recall_curve(self, eval_manager, rf_classifier, classification_data):
        """Test the plot_precision_recall_curve method of EvaluationManager."""
        X, y, _ = classification_data
        filename = "precision_recall_curve"

        with patch("os.makedirs") as mock_makedirs, \
            patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager.plot_precision_recall_curve(rf_classifier, X, y, filename, pos_label="A")

            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_calc_plot_precision_recall_curve(self, eval_manager, classification_data, classification100_data, classification100_group_data):
         # classification.csv
        _, y, _ = classification_data
        y_score = np.array([
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
            [0.8, 0.2],
        ])[:, 1]
        plot_data, ap_score = eval_manager._calc_plot_precision_recall_curve(y_score, y, pos_label="A")
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (6, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["Recall"].tolist()[i], [1.0, 0.666667, 0.333333, 0.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["Precision"].tolist()[i], [0.75, 1.0, 1.0, 1.0, 0.916667, 0.916667][i])
            assert plot_data["Type"].tolist()[i] == ["PR Curve", "PR Curve", "PR Curve", "PR Curve", "AP Score = 0.92", "AP Score = 0.92"][i]
        assert np.isclose(ap_score, 0.9166666666)

        # classification100.csv
        _, y, _ = classification100_data
        y_score = np.array([
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
        ] * 8)[:, 1]
        plot_data, ap_score = eval_manager._calc_plot_precision_recall_curve(y_score, y, pos_label=0)
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (6, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["Recall"].tolist()[i], [1.0, 0.650, 0.425, 0.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["Precision"].tolist()[i], [0.5, 0.541667, 0.708333, 1.0, 0.597917, 0.597917][i])
            assert plot_data["Type"].tolist()[i] == ["PR Curve", "PR Curve", "PR Curve", "PR Curve", "AP Score = 0.60", "AP Score = 0.60"][i]
        assert np.isclose(ap_score, 0.5979166666666667)

        # classification100_group.csv
        _, y, _ = classification100_group_data
        y_score = np.array([
            [0.8, 0.2],
            [0.55, 0.45],
            [0.1, 0.9],
        ] * 17)[:, 1]
        plot_data, ap_score = eval_manager._calc_plot_precision_recall_curve(y_score, y, pos_label=0)
        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape == (6, 3)
        for i in range(len(plot_data)):
            assert np.isclose(plot_data["Recall"].tolist()[i], [1.0, 0.666667, 0.333333, 0.0, 0.0, 1.0][i])
            assert np.isclose(plot_data["Precision"].tolist()[i], [0.529412, 0.529412, 0.529412, 1.0, 0.529412, 00.529412][i])
            assert plot_data["Type"].tolist()[i] == ["PR Curve", "PR Curve", "PR Curve", "PR Curve", "AP Score = 0.53", "AP Score = 0.53"][i]
        assert np.isclose(ap_score, 0.5294117647058822)

    def test_save_to_json(self, eval_manager, tmpdir):
        data = {"score": 0.95}
        metadata = {"experiment": "test"}
        output_path = tmpdir.join("test_result.json")

        with patch("builtins.open", mock_open()) as mock_file:
            eval_manager._save_to_json(data, str(output_path), metadata)

            # Verify that the file was opened and json was written
            mock_file.assert_called_once_with(
                str(output_path), "w", encoding="utf-8"
                )
            
            # Check that metadata was added to the data dictionary
            data_with_metadata = data.copy()
            data_with_metadata["_metadata"] = metadata

            # Retrieve the actual JSON written by combining all `write` calls
            written_data = "".join(
                call.args[0] for call in mock_file().write.call_args_list
                )

            # Convert JSON string back to a dictionary for comparison
            written_data_dict = json.loads(written_data)
            assert written_data_dict == data_with_metadata

    def test_save_to_json_io_error(self, eval_manager, tmpdir):
        data = {"score": 0.95}
        metadata = {"experiment": "test"}
        output_path = tmpdir.join("test_result.json")

        # Simulate IOError
        with patch("builtins.open", side_effect=IOError("File write error")) as mock_file:
            eval_manager._save_to_json(data, str(output_path), metadata)

            # Verify that logger.info was called due to IOError
            eval_manager.logger.info.assert_called_once_with(
                f"Failed to save JSON to {str(output_path)}: File write error"
            )

    def test_save_plot(self, eval_manager, tmpdir):
        output_path = tmpdir.join("test_plot.png")
        metadata = {"experiment": "test"}

        # Create a simple plot to test saving
        plt.plot([1, 2, 3], [4, 5, 6])

        with patch("matplotlib.pyplot.savefig") as mock_savefig, \
             patch("matplotlib.pyplot.close") as mock_close:
            eval_manager._save_plot(str(output_path), metadata)

            # Assert savefig was called with the correct parameters
            mock_savefig.assert_called_once_with(
                str(output_path), format="png", metadata=metadata
                )
            mock_close.assert_called_once()

    def test_save_plot_io_error(self, eval_manager, tmpdir):
        output_path = tmpdir.join("test_plot.png")
        metadata = {"experiment": "test"}

        plt.plot([1, 2, 3], [4, 5, 6])

        # Simulate IOError in plt.savefig
        with patch("matplotlib.pyplot.savefig", side_effect=IOError("File write error")):
            eval_manager._save_plot(str(output_path), metadata)

            # Verify that logger.info was called due to IOError
            eval_manager.logger.info.assert_called_once_with(
                f"Failed to save plot to {str(output_path)}: File write error"
            )

    def test_save_and_load_model(self, eval_manager, rf_regressor, tmpdir):
        filename = tmpdir.join("saved_model")
        eval_manager.output_dir = tmpdir

        with patch("os.makedirs") as mock_makedirs:
            eval_manager.save_model(rf_regressor, filename)
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            
        assert Path(f"{filename}.pkl").exists()

        loaded_model = eval_manager.load_model(f"{filename}.pkl")
        assert isinstance(loaded_model["model"], RandomForestRegressor)

    def test_get_metadata(self, eval_manager, rf_regressor, ridge):
        metadata = eval_manager._get_metadata(rf_regressor)
        assert isinstance(metadata, dict)
        assert metadata["timestamp"] is not None
        assert isinstance(metadata["method"], str)
        assert metadata["method"] == "test_get_metadata"
        assert metadata["models"] == {"rf": "Random Forest"}
        assert metadata["is_test"] == "False"

        metadata = eval_manager._get_metadata(ridge, "another_method", is_test=True)
        assert isinstance(metadata, dict)
        assert metadata["timestamp"] is not None
        assert isinstance(metadata["method"], str)
        assert metadata["method"] == "another_method"
        assert metadata["models"] == {"ridge": "Ridge Regression"}
        assert metadata["is_test"] == "True"

    def test_get_group_index(self, eval_manager):
        group_index = eval_manager._get_group_index(True)
        assert group_index is None

        group_index = eval_manager._get_group_index(False)
        assert group_index is None

        eval_manager.data_has_groups = True
        eval_manager.group_index_train = {
            "values": np.array([1, 2, 3]),
            "indices": np.array([0, 1, 2]),
            "series": pd.Series([1, 2, 3])
        }
        eval_manager.group_index_test = {
            "values": np.array([4, 5, 6]),
            "indices": np.array([3, 4, 5]),
            "series": pd.Series([4, 5, 6])
        }
        group_index = eval_manager._get_group_index(False)
        assert np.array_equal(group_index["values"], np.array([1, 2, 3]))
        assert np.array_equal(group_index["indices"], np.array([0, 1, 2]))
        assert pd.Series(group_index["series"]).equals(pd.Series([1, 2, 3]))

        group_index = eval_manager._get_group_index(True)
        assert np.array_equal(group_index["values"], np.array([4, 5, 6]))
        assert np.array_equal(group_index["indices"], np.array([3, 4, 5]))
        assert pd.Series(group_index["series"]).equals(pd.Series([4, 5, 6]))

    def test_get_cv_splitter(self, eval_manager, regression100_data, regression100_group_data):
        _, y, _ = regression100_data
        y_categorical = pd.Series([1, 0] * 25) 
        y_categorical.attrs["is_test"] = False

        splitter, _ = eval_manager._get_cv_splitter(y, 5, None)
        assert isinstance(splitter, model_selection.KFold)

        splitter, _ = eval_manager._get_cv_splitter(y, 5, 2)
        assert isinstance(splitter, model_selection.RepeatedKFold)

        splitter, _ = eval_manager._get_cv_splitter(
            y_categorical, 5, None
        )
        assert isinstance(splitter, model_selection.StratifiedKFold)

        splitter, _ = eval_manager._get_cv_splitter(
            y_categorical, 5, 2
        )
        assert isinstance(splitter, model_selection.RepeatedStratifiedKFold)

        eval_manager.data_has_groups = True
        eval_manager.group_index_train = {
            "values": np.array([1, 2, 3]),
            "indices": np.array([0, 1, 2]),
            "series": pd.Series([1, 2, 3])
        }
        eval_manager.group_index_test = {
            "values": np.array([4, 5, 6]),
            "indices": np.array([3, 4, 5]),
            "series": pd.Series([4, 5, 6])
        }
        splitter, _ = eval_manager._get_cv_splitter(y, 5, None)
        assert isinstance(splitter, model_selection.GroupKFold)

        splitter, _ = eval_manager._get_cv_splitter(y, 5, 2)
        assert isinstance(splitter, model_selection.GroupKFold)

        splitter, _ = eval_manager._get_cv_splitter(
            y_categorical, 5, None
        )
        assert isinstance(splitter, model_selection.StratifiedGroupKFold)

        splitter, _ = eval_manager._get_cv_splitter(
            y_categorical, 5, 2
        )
        assert isinstance(splitter, model_selection.StratifiedGroupKFold)
