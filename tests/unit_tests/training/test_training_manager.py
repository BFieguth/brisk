"""Unit tests for the TrainingManager class."""

# pylint: disable=protected-access, redefined-outer-name, inconsistent-quotes
# NOTE inconsistent quotes are due to the use of single quotes in some f-strings
# Double quotes can be used in Python 3.12, but not in Python 3.11 or 3.10.

import collections
import datetime
import importlib
import logging
import os
import re
import sys

import pandas as pd
import pytest
from unittest import mock
from sklearn import linear_model, preprocessing

from brisk.data.data_manager import DataManager
from brisk.data.data_split_info import DataSplitInfo
from brisk.utility.algorithm_wrapper import AlgorithmWrapper
from brisk.evaluation.metric_manager import MetricManager
from brisk.training.training_manager import TrainingManager
from brisk.configuration.experiment import Experiment
from brisk.training.workflow import Workflow
from brisk.version import __version__

@pytest.fixture
def metric_config(mock_regression_project):
    metric_file = mock_regression_project / "metric.py"
    spec = importlib.util.spec_from_file_location("metric", str(metric_file))
    metric_module = importlib.util.module_from_spec(spec)
    sys.modules["metric"] = metric_module
    spec.loader.exec_module(metric_module)

    return metric_module.METRIC_CONFIG


@pytest.fixture
def configuration(mock_regression_project):
    """
    Create a configuration instance using the create_configuration function.
    """
    settings_file = mock_regression_project / "settings.py"
    spec = importlib.util.spec_from_file_location(
        "settings", str(settings_file)
    )
    settings_module = importlib.util.module_from_spec(spec)
    sys.modules["settings"] = settings_module
    spec.loader.exec_module(settings_module)

    return settings_module.create_configuration()


@pytest.fixture
def test_workflow(mock_regression_project):
    workflow_file = mock_regression_project / "workflows" / "test_workflow.py"
    spec = importlib.util.spec_from_file_location(
        "test_workflow", str(workflow_file)
    )
    workflow_module = importlib.util.module_from_spec(spec)
    sys.modules["test_workflow"] = workflow_module
    spec.loader.exec_module(workflow_module)

    return workflow_module.Regression


@pytest.fixture
def training_manager(metric_config, configuration):
    return TrainingManager(metric_config, configuration)


class TestTrainingManager:
    """Unit tests for the TrainingManager class."""
    def test_init(self, training_manager):
        assert isinstance(training_manager.metric_config, MetricManager)

        assert isinstance(training_manager.verbose, bool)
        assert training_manager.verbose is False

        assert isinstance(training_manager.data_managers, dict)
        for _, value in training_manager.data_managers.items():
            assert isinstance(value, DataManager)

        assert isinstance(training_manager.experiments, collections.deque)
        for value in training_manager.experiments:
            assert isinstance(value, Experiment)

        assert isinstance(training_manager.logfile, str)

        assert isinstance(training_manager.output_structure, dict)

        assert isinstance(training_manager.description_map, dict)

        assert isinstance(
            training_manager.experiment_paths, collections.defaultdict
        )
        assert isinstance(
            training_manager.experiment_paths[0], collections.defaultdict
        )
        assert isinstance(training_manager.experiment_paths[0][0], dict)

        assert isinstance(
            training_manager.experiment_results, collections.defaultdict
        )
        assert isinstance(
            training_manager.experiment_results[0], collections.defaultdict
        )
        assert isinstance(training_manager.experiment_results[0][0], list)

    @mock.patch("brisk.training.training_manager.datetime")
    def test_create_results_dir_with_timestamp(
        self,
        mock_datetime,
        training_manager
    ):
        mock_datetime.now.return_value = datetime.datetime(2024, 1, 1, 12, 0, 0)
        expected_dir = os.path.join("results", "01_01_2024_12_00_00")
        results_dir = training_manager._create_results_dir(None)

        assert os.path.normpath(results_dir) == os.path.normpath(expected_dir)
        assert os.path.exists(results_dir)

    def test_create_results_dir_with_custom_name(self, training_manager):
        custom_name = "test_results"
        results_dir = training_manager._create_results_dir(custom_name)
        expected_dir = os.path.join("results", custom_name)

        assert results_dir == expected_dir
        assert os.path.exists(results_dir)

    def test_create_results_dir_already_exists(self, training_manager):
        existing_dir = os.path.normpath(
            os.path.join("results", "existing_results")
        )
        os.makedirs(existing_dir)

        with pytest.raises(
            FileExistsError,
            match=f"Results directory '{re.escape(existing_dir)}' already exists." # pylint: disable=line-too-long
        ):
            training_manager._create_results_dir("existing_results")

    def test_save_data_distribution(self, tmp_path, training_manager):
        class DataManagerNoScaler:
            def split(
                    self,
                    data_path,
                    categorical_features,
                    table_name,
                    group_name,
                    filename
                ): # pylint: disable=unused-argument
                split_info = DataSplitInfo(
                    X_train=pd.DataFrame([[1, 2], [3, 4]]),
                    X_test=pd.DataFrame([[5, 6]]),
                    y_train=pd.Series([0, 1]),
                    y_test=pd.Series([1]),
                    filename=filename,
                    scaler=None,
                    features=["feature1", "feature2"]
                )
                return split_info


        class DataManagerWithScaler(DataManagerNoScaler):
            def split(
                self,
                data_path,
                categorical_features,
                table_name,
                group_name,
                filename
            ):
                split_info = super().split(
                    data_path, categorical_features, table_name, group_name,
                    filename
                )
                split_info.scaler = preprocessing.StandardScaler()
                return split_info


        training_manager.data_managers = {
            "group1": DataManagerNoScaler(),
            "group2": DataManagerWithScaler()
        }

        output_structure = {
            "group1": {
                "dataset1": ("path/to/data1.csv", "group1"),
                "dataset2": ("path/to/data2.csv", "group1")
            },
            "group2": {
                "dataset3": ("path/to/data3.csv", "group2")
            }
        }

        with mock.patch("json.dump"):
            training_manager._save_data_distributions(
                str(tmp_path), output_structure
            )

        # Assert directory structure
        assert (tmp_path / "group1").exists()
        assert (tmp_path / "group1" / "dataset1").exists()
        assert (tmp_path / "group1" / "dataset2").exists()
        assert (
            tmp_path / "group1" / "dataset1" / "split_distribution"
        ).exists()

        assert (
            tmp_path / "group1" / "dataset2" / "split_distribution"
        ).exists()

        assert not list(
            (tmp_path / "group1" / "dataset1").glob("*.joblib")
            ), "No .joblib files should exist in dataset1 for group1."

        assert not list(
            (tmp_path / "group1" / "dataset2").glob("*.joblib")
            ), "No .joblib files should exist in dataset2 for group1."

        assert (tmp_path / "group2").exists()
        assert (tmp_path / "group2" / "dataset3").exists()
        assert (
            tmp_path / "group2" / "dataset3" / "split_distribution"
        ).exists()

        scaler_path = (
            tmp_path / "group2" / "dataset3" / "dataset3_StandardScaler.joblib"
        )
        assert scaler_path.exists(), f"Scaler file {scaler_path} doesn't exist."

    def test_setup_logger_verbose(self, tmp_path, training_manager):
        for logger in logging.Logger.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                logger.handlers.clear()

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        training_manager.verbose = True
        logger = training_manager._setup_logger(results_dir)

        assert logger.name == "TrainingManager"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 2
        assert logger.handlers[0].level == logging.WARNING
        assert logger.handlers[1].level == logging.INFO

    def test_setup_logger_not_verbose(self, tmp_path, training_manager):
        for logger in logging.Logger.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                logger.handlers.clear()

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        training_manager.verbose = False
        logger = training_manager._setup_logger(results_dir)

        assert logger.name == "TrainingManager"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 2
        assert logger.handlers[0].level == logging.WARNING
        assert logger.handlers[1].level == logging.ERROR

    def test_setup_workflow(self, test_workflow, training_manager):
        training_manager.logger = mock.Mock()

        group_name = "test_group"
        current_experiment = Experiment(
            group_name=group_name,
            dataset_path="./datasets/data.csv",
            algorithms={"model": AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression
            )},
            table_name=None,
            categorical_features=None
        )
        workflow = test_workflow
        results_dir = ""
        dataset_name = "data"
        experiment_name = current_experiment.name
        expected_algo_names = ["linear"]

        workflow_instance = training_manager._setup_workflow(
            current_experiment, workflow, results_dir, group_name, dataset_name,
            experiment_name
        )

        assert isinstance(workflow_instance, Workflow)
        assert workflow_instance.algorithm_names == expected_algo_names
        assert isinstance(
            workflow_instance.model, linear_model.LinearRegression
        )
        assert (training_manager.experiment_paths
                [group_name]
                [dataset_name]
                [experiment_name]) == os.path.join(
                    results_dir, group_name, dataset_name, experiment_name
                )

    @mock.patch("tqdm.tqdm.write")
    @mock.patch("time.time")
    def test_handle_success(self, mock_time, mock_write, training_manager):
        mock_time.return_value = 1734371288

        start_time = 1734371100
        group_name = "group1"
        dataset_name = "dataset1"
        experiment_name = "experiment1"
        expected_tqdm_calls = (
            "\nExperiment 'experiment1' on dataset 'dataset1' "
            "PASSED in 3m 8s.",
            f"\n{'-' * 80}"
        )

        training_manager._handle_success(
            start_time, group_name, dataset_name, experiment_name
        )
        assert (training_manager.experiment_results
                [group_name][dataset_name][-1]
            ) == {
                "experiment": "experiment1",
                "status": "PASSED",
                "time_taken": "3m 8s"
            }
        for expected_message in expected_tqdm_calls:
            mock_write.assert_any_call(expected_message)

    @mock.patch("tqdm.tqdm.write")
    @mock.patch("time.time")
    def test_handle_failure(self, mock_time, mock_write, training_manager):
        mock_time.return_value = 1734371288
        training_manager.logger = mock.Mock()

        group_name = "group1"
        dataset_name = "dataset1"
        experiment_name = "experiment1"
        start_time = 1734371000
        error = "This is a test error"
        expected_error_message = (
            "\n\nDataset Name: dataset1\n"
            "Experiment Name: experiment1\n\n"
            "Error: This is a test error"
        )
        expected_tqdm_calls = (
            "\nExperiment 'experiment1' on dataset 'dataset1' "
            "FAILED in 4m 48s.",
            f"\n{'-' * 80}"
        )

        training_manager._handle_failure(
            group_name, dataset_name, experiment_name, start_time, error
        )

        training_manager.logger.exception.assert_called_once_with(
            expected_error_message
        )
        assert (training_manager.experiment_results
                [group_name][dataset_name][-1]
            ) == {
                "experiment": "experiment1",
                "status": "FAILED",
                "time_taken": "4m 48s",
                "error": "This is a test error"
            }
        for expected_message in expected_tqdm_calls:
            mock_write.assert_any_call(expected_message)

    @mock.patch("brisk.training.training_manager.logging.getLogger")
    def test_log_warning(self, mock_get_logger, training_manager):
        mock_logger = mock.MagicMock()
        mock_get_logger.return_value = mock_logger

        message = "This is a test warning"
        category = UserWarning
        filename = "test_file.py"
        lineno = 42
        dataset_name = "test_dataset"
        experiment_name = "test_experiment"
        expected_log_message = (
            "\n\nDataset Name: test_dataset \n"
            "Experiment Name: test_experiment\n\n"
            "Warning in test_file.py at line 42:\n"
            "Category: UserWarning\n\n"
            "Message: This is a test warning\n"
        )

        training_manager._log_warning(
            message, category, filename, lineno, dataset_name, experiment_name
        )

        mock_get_logger.assert_called_once_with("TrainingManager")
        mock_logger.warning.assert_called_once()
        mock_logger.warning.assert_called_with(expected_log_message)

    @mock.patch("builtins.print")
    def test_print_experiment_summary(self, mock_print, training_manager):
        training_manager.experiment_results = {
            "group1": {
                "dataset1": [
                    {
                        "experiment": "exp1", 
                        "status": "PASSED", 
                        "time_taken": "1m 30s"
                    },
                    {
                        "experiment": "exp2", 
                        "status": "FAILED", 
                        "time_taken": "2m 15s"
                    },
                ],
                "dataset2": [
                    {
                        "experiment": "exp3",
                        "status": "PASSED",
                        "time_taken": "3m 45s"
                    },
                ],
            },
            "group2": {
                "dataset3": [
                    {
                        "experiment": "exp4",
                        "status": "PASSED", 
                        "time_taken": "4m 5s"
                    },
                ],
            },
        }

        training_manager._print_experiment_summary()

        expected_calls = [
            mock.call("\n" + "="*70),
            mock.call("EXPERIMENT SUMMARY"),
            mock.call("="*70),
            mock.call("\nGroup: group1"),
            mock.call("="*70),
            mock.call("\nDataset: dataset1"),
            mock.call(f"{'Experiment':<50} {'Status':<10} {'Time':<10}"),
            mock.call("-"*70),
            mock.call(f"{'exp1':<50} {'PASSED':<10} {'1m 30s':<10}"),
            mock.call(f"{'exp2':<50} {'FAILED':<10} {'2m 15s':<10}"),
            mock.call("\nDataset: dataset2"),
            mock.call(f"{'Experiment':<50} {'Status':<10} {'Time':<10}"),
            mock.call("-"*70),
            mock.call(f"{'exp3':<50} {'PASSED':<10} {'3m 45s':<10}"),
            mock.call("="*70),
            mock.call("\nGroup: group2"),
            mock.call("="*70),
            mock.call("\nDataset: dataset3"),
            mock.call(f"{'Experiment':<50} {'Status':<10} {'Time':<10}"),
            mock.call("-"*70),
            mock.call(f"{'exp4':<50} {'PASSED':<10} {'4m 5s':<10}"),
            mock.call("="*70),
            mock.call("\nModels trained using Brisk version", __version__)
        ]

        assert mock_print.call_args_list == expected_calls

    @mock.patch("brisk.training.training_manager.os.path.normpath")
    @mock.patch("brisk.training.training_manager.os.path.exists")
    @mock.patch("brisk.training.training_manager.os.makedirs")
    @mock.patch("brisk.training.training_manager.os.path.join")
    def test_get_experiment_dir(
        self,
        mock_join,
        mock_makedirs,
        mock_exists,
        mock_normpath,
        training_manager,
    ):
        results_dir = "test_results"
        group_name = "test_group"
        dataset_name = "data"
        experiment_name = "experiment"
        mock_exists.return_value = False

        training_manager._get_experiment_dir(
            results_dir, group_name, dataset_name, experiment_name
        )

        mock_normpath.assert_called_once()
        mock_join.assert_called_once()
        mock_exists.assert_called_once()
        mock_makedirs.assert_called_once()

    @mock.patch("brisk.training.training_manager.os.path.normpath")
    @mock.patch("brisk.training.training_manager.os.path.exists")
    @mock.patch("brisk.training.training_manager.os.makedirs")
    @mock.patch("brisk.training.training_manager.os.path.join")
    def test_get_experiment_dir_exists(
        self,
        mock_join,
        mock_makedirs,
        mock_exists,
        mock_normpath,
        training_manager,
    ):
        results_dir = "test_results"
        group_name = "test_group"
        dataset_name = "data"
        experiment_name = "experiment"
        mock_exists.return_value = True

        training_manager._get_experiment_dir(
            results_dir, group_name, dataset_name, experiment_name
        )

        mock_normpath.assert_called_once()
        mock_join.assert_called_once()
        mock_exists.assert_called_once()
        mock_makedirs.assert_not_called()

    def test_format_time(self, training_manager):
        result = training_manager._format_time(0)
        assert result == "0m 0s"

        result = training_manager._format_time(59)
        assert result == "0m 59s"

        result = training_manager._format_time(60)
        assert result == "1m 0s"

        result = training_manager._format_time(125)
        assert result == "2m 5s"

        result = training_manager._format_time(3661)
        assert result == "61m 1s"

    @mock.patch("brisk.training.training_manager.logging.shutdown")
    @mock.patch("brisk.training.training_manager.os.path.exists")
    @mock.patch("brisk.training.training_manager.os.remove")
    @mock.patch("brisk.training.training_manager.os.path.getsize")
    def test_cleanup_empty_error_log(
        self,
        mock_getsize,
        mock_remove,
        mock_exists,
        mock_logging_shutdown,
        training_manager,
    ):
        results_dir = "test_results"
        progress_bar = mock.MagicMock()

        mock_exists.return_value = True
        mock_getsize.return_value = 0

        training_manager._cleanup(results_dir, progress_bar)

        progress_bar.close.assert_called_once()
        mock_logging_shutdown.assert_called_once()
        mock_exists.assert_called_once_with(
            os.path.join(results_dir, "error_log.txt")
        )
        mock_getsize.assert_called_once_with(
            os.path.join(results_dir, "error_log.txt")
        )
        mock_remove.assert_called_once_with(
            os.path.join(results_dir, "error_log.txt")
        )

    @mock.patch("brisk.training.training_manager.logging.shutdown")
    @mock.patch("brisk.training.training_manager.os.path.exists")
    @mock.patch("brisk.training.training_manager.os.remove")
    @mock.patch("brisk.training.training_manager.os.path.getsize")
    def test_cleanup_non_empty_error_log(
        self,
        mock_getsize,
        mock_remove,
        mock_exists,
        mock_logging_shutdown,
        training_manager,
    ):
        results_dir = "test_results"
        progress_bar = mock.MagicMock()

        mock_exists.return_value = True
        mock_getsize.return_value = 100

        training_manager._cleanup(results_dir, progress_bar)

        progress_bar.close.assert_called_once()
        mock_logging_shutdown.assert_called_once()
        mock_exists.assert_called_once_with(
            os.path.join(results_dir, "error_log.txt")
        )
        mock_getsize.assert_called_once_with(
            os.path.join(results_dir, "error_log.txt")
        )
        mock_remove.assert_not_called()
