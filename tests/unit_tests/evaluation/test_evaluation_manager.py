from pathlib import Path
from unittest.mock import patch
from importlib import util
import os
from unittest import mock

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from brisk.evaluation.evaluation_manager import EvaluationManager
from brisk.evaluation.metric_manager import MetricManager
from brisk.evaluation.evaluators.registry import EvaluatorRegistry
from brisk.services import GlobalServiceManager
from brisk.evaluation.evaluators.base import BaseEvaluator
from brisk.services.bundle import ServiceBundle
from brisk.services.utility import UtilityService
from brisk.services.metadata import MetadataService
from brisk.theme.plot_settings import PlotSettings

@pytest.fixture
def data_manager(mock_brisk_project, tmp_path, mock_services):
    with mock.patch("brisk.data.data_manager.get_services", return_value=mock_services):
        data_file = tmp_path / "data.py"
        spec = util.spec_from_file_location("data", data_file)
        data_module = util.module_from_spec(spec)
        spec.loader.exec_module(data_module)
    return data_module.BASE_DATA_MANAGER


@pytest.fixture
def algorithm_config(mock_brisk_project, tmp_path):
    algorithm_file = tmp_path / "algorithms.py"
    spec = util.spec_from_file_location("algorithms", algorithm_file)
    algorithm_module = util.module_from_spec(spec)
    spec.loader.exec_module(algorithm_module)
    return algorithm_module.ALGORITHM_CONFIG


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
    services.metadata = MetadataService("metadata", algorithm_config)
    services.utility.set_plot_settings(PlotSettings())
    services.reporting = mock.MagicMock()
    services.reporting.add_dataset = mock.MagicMock()
    return services


@pytest.fixture
def regression_data(data_manager, tmp_path, mock_services):
    with mock.patch("brisk.data.data_split_info.get_services", return_value=mock_services):
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
def eval_manager(tmp_path, mock_brisk_project, mock_services):
    with mock.patch("brisk.evaluation.evaluation_manager.get_services", return_value=mock_services):
        metric_file = tmp_path / "metrics.py"
        spec = util.spec_from_file_location("metrics", metric_file)
        metric_module = util.module_from_spec(spec)
        spec.loader.exec_module(metric_module)
        metric_config = metric_module.METRIC_CONFIG
        return EvaluationManager(metric_config)


@pytest.fixture
def rf_regressor(algorithm_config, regression_data):
    X, y, _= regression_data
    model = algorithm_config["rf"].instantiate()
    model.fit(X, y)
    return model


class TestEvaluationManager:
    def test_init(self, eval_manager):
        assert isinstance(eval_manager, EvaluationManager)
        assert isinstance(eval_manager.metric_config, MetricManager)
        assert eval_manager.output_dir is None
        assert isinstance(eval_manager.registry, EvaluatorRegistry)

    def test_set_experiment_values(self, eval_manager, tmp_path):
        output_dir = os.path.join(
            tmp_path, "test", "path" 
        )
        split_metadata = {
            "method": "test_algo",
            "is_test": "True"
        }
        group_index_train = {"feature": np.array([1, 2, 3, 4, 5])}
        group_index_test = {"feature": np.array([6, 7, 8])}
        eval_manager.set_experiment_values(
            output_dir=output_dir,
            split_metadata=split_metadata,
            group_index_train=group_index_train,
            group_index_test=group_index_test
        )
        assert isinstance(eval_manager.output_dir, Path)
        assert GlobalServiceManager.instance.services["io"].output_dir == Path(output_dir)

    def test_update_metrics(self, eval_manager):
        wrapper = eval_manager.metric_config._metrics_by_name["r2_score"]
        assert wrapper.params["split_metadata"] == {}

        split_metadata = {"num_features": 5, "num_samples": 20}
        eval_manager.update_metrics(split_metadata)
        wrapper = eval_manager.metric_config._metrics_by_name["r2_score"]
        assert wrapper.params["split_metadata"] == split_metadata

    def test_get_evaluator(self, eval_manager):
        evaluator = eval_manager.get_evaluator("brisk_evaluate_model")
        assert isinstance(evaluator, BaseEvaluator)
        assert evaluator.metric_config == eval_manager.metric_config

    def test_save_and_load_model(self, eval_manager, rf_regressor, tmpdir):
        filename = tmpdir.join("saved_model")
        eval_manager.output_dir = tmpdir

        with (
            patch("os.makedirs") as mock_makedirs,
        ):
            eval_manager.save_model(rf_regressor, filename)
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)

        assert Path(f"{filename}.pkl").exists()

        loaded_model = eval_manager.load_model(f"{filename}.pkl")
        assert isinstance(loaded_model["model"], RandomForestRegressor)

    def test_register_custom_evaluators(self, eval_manager, monkeypatch):
        # Check classes from evaluators.py get registered   
        assert "registered_class" in eval_manager.registry.evaluators
