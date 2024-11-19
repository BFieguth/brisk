import datetime
import os
import copy
from unittest.mock import patch, MagicMock, Mock, create_autospec
from collections import deque
import itertools
import logging
import warnings

import pandas as pd
import pytest

from brisk.training.TrainingManager import TrainingManager
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper
from brisk.data.DataSplitInfo import DataSplitInfo
from brisk.training.Workflow import Workflow

class MockDataManager:
    def split(self, data_path, table_name):
        X_train = pd.DataFrame([[1, 2], [3, 4]])
        X_test = pd.DataFrame([[5, 6]])
        y_train = pd.Series([0, 1])
        y_test = pd.Series([1])
        scaler = None
        feature_names = ['feature1', 'feature2']
        return X_train, X_test, y_train, y_test, scaler, feature_names

    @property
    def categorical_features(self):
        return ['feature1']


class MockEvaluationManager:
    def __init__(self, algorithm_config, metric_config):
        self.algorithm_config = algorithm_config
        self.metric_config = metric_config
   

class MockDataSplitInfo:
    def __init__(
        self, 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        filename, 
        scaler, 
        features, 
        categorical_features
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.filename = filename
        self.scaler = scaler
        self.features = features
        self.categorical_features = categorical_features

    def save_distribution(self, dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    def get_train_test(self):
        """Return training and test data."""
        return self.X_train, self.X_test, self.y_train, self.y_test


class TestTrainingManager:

    @pytest.fixture
    def algorithm_config(self):
        return [
            AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=Mock()
            ),
            AlgorithmWrapper(
                name="ridge",
                display_name="Ridge Regression",
                algorithm_class=Mock(),
                default_params={"max_iter": 10000},
                hyperparam_grid={"alpha": [0.1, 0.2]}
            ),
        ]
    
    @pytest.fixture
    def metric_config(self):
        return {
            'explained_variance_score': {
                'display_name': 'Explained Variance Score',
                'func': MagicMock(),
                'scorer': MagicMock(),
            },
            'max_error': {
                'display_name': 'Max Error',
                'func': MagicMock(),
                'scorer': MagicMock(),
            },
            'mean_absolute_error': {
                'display_name': 'Mean Absolute Error',
                'abbr': 'MAE',
                'func': MagicMock(),
                'scorer': MagicMock(),
            },
        }
    
    @pytest.fixture
    def algorithms(self):
        return ['linear', 'ridge']

    @pytest.fixture
    def data_paths(self):
        return [('data1.csv', ''), ('data2.csv', '')]

    @pytest.fixture
    def training_manager(self, algorithm_config, metric_config, algorithms, data_paths):
        with patch('brisk.training.TrainingManager.EvaluationManager', MockEvaluationManager), \
             patch('brisk.training.TrainingManager.DataSplitInfo', MockDataSplitInfo):
            tm = TrainingManager(
                algorithm_config=algorithm_config,
                metric_config=metric_config,
                data_manager=MockDataManager(),
                algorithms=algorithms,
                data_paths=data_paths,
                verbose=False
            )
        return tm
    
    @pytest.fixture
    def mock_save_report(self):
        """Mock ReportManager for creating reports."""
        with patch("brisk.reporting.ReportManager.ReportManager") as MockReportManager:
            yield MockReportManager
    
    def test_init(self, training_manager, algorithm_config, metric_config, algorithms, data_paths):
        """Test the __init__ method."""
        assert training_manager.algorithm_config == algorithm_config
        assert training_manager.metric_config == metric_config
        assert isinstance(training_manager.DataManager, MockDataManager)
        assert training_manager.algorithms == algorithms
        assert training_manager.data_paths == data_paths
        assert hasattr(training_manager, 'data_splits')
        assert hasattr(training_manager, 'experiments')

    def test_validate_methods_valid(self, training_manager):
        """Test _validate_algorithms with valid algorithms."""
        training_manager._validate_algorithms()

    def test_validate_methods_invalid(self, algorithm_config, metric_config, data_paths):
        """Test _validate_algorithms with invalid algorithms."""
        invalid_methods = ['linear', 'invalid_method']
        with pytest.raises(ValueError) as exc_info:
            with patch('brisk.training.TrainingManager.EvaluationManager', MockEvaluationManager), \
                 patch('brisk.data.DataSplitInfo.DataSplitInfo', MockDataSplitInfo):
                TrainingManager(
                    algorithm_config=algorithm_config,
                    metric_config=metric_config,
                    data_manager=MockDataManager(),
                    algorithms=invalid_methods,
                    data_paths=data_paths,
                    verbose=False
                )
        assert "The following algorithms are not included in the configuration: ['invalid_method']" in str(exc_info.value)

    def test_get_data_splits(self, training_manager):
        """Test _get_data_splits method."""
        data_splits = training_manager._get_data_splits()
        assert isinstance(data_splits, dict)
        for data_path, data_split in data_splits.items():
            assert isinstance(data_split, DataSplitInfo)
            assert not data_split.scaler

    def test_create_experiments(self, training_manager):
        """Test _create_experiments method."""
        experiments = training_manager._create_experiments()
        method_combinations = [('linear',), ('ridge',)]
        expected_experiments = deque(itertools.product(
            training_manager.data_paths,
            method_combinations
        ))
        assert list(experiments) == list(expected_experiments)

    def test_get_results_dir(self, training_manager):
        """Test _get_results_dir method."""
        results_dir = training_manager._get_results_dir()
        assert results_dir.startswith("results")
        timestamp_format = "%d_%m_%Y_%H_%M_%S"
        timestamp_str = results_dir.split(os.sep)[-1]
        datetime.datetime.strptime(timestamp_str, timestamp_format)  # Should not raise exception

    def test_get_experiment_dir(self, training_manager):
        """Test _get_experiment_dir method."""
        mock_results_dir = "/mock/results"

        with patch.object(training_manager, "_get_results_dir", return_value=mock_results_dir), \
            patch("os.makedirs") as mock_makedirs:
            method_name = 'linear'
            data_path = ('data1.csv', '')
            experiment_dir = training_manager._get_experiment_dir(
                method_name, data_path, mock_results_dir
            )
            
            # Use os.path.normpath to normalize path separators for the OS
            expected_dir = os.path.normpath(os.path.join(mock_results_dir, 'data1', method_name))
            experiment_dir = os.path.normpath(experiment_dir)
            assert experiment_dir == expected_dir
            
            # Verify that os.makedirs was called with the expected directory
            mock_makedirs.assert_called_once_with(expected_dir)

    def test_save_scalers(self, training_manager, tmpdir):
        """Test _save_scalers method."""
        results_dir = tmpdir.mkdir("results")
        training_manager._save_scalers(str(results_dir))
        scaler_dir = os.path.join(str(results_dir), "scalers")
        assert os.path.exists(scaler_dir)
        # Since our mock scalers are None, there should be no files
        assert len(os.listdir(scaler_dir)) == 0

    def test_save_data_distributions(self, training_manager, tmpdir):
        """Test _save_data_distributions method."""
        results_dir = tmpdir.mkdir("results")
        training_manager._save_data_distributions(str(results_dir))
        for data_split in training_manager.data_splits.values():
            dataset_dir = os.path.join(str(results_dir), data_split.filename, "feature_distribution")
            assert os.path.exists(dataset_dir)

    def test_print_experiment_summary(self, capsys, training_manager):
        """Test _print_experiment_summary method."""
        experiment_results = {
            'dataset1': [
                {'experiment': 'exp1', 'status': 'Success', 'time_taken': '00:01'},
                {'experiment': 'exp2', 'status': 'Failed', 'time_taken': '00:02'},
            ]
        }
        training_manager._print_experiment_summary(experiment_results)
        captured = capsys.readouterr()
        assert "EXPERIMENT SUMMARY" in captured.out
        assert "Dataset: dataset1" in captured.out
        assert "exp1" in captured.out
        assert "Success" in captured.out
        assert "00:01" in captured.out
        assert "exp2" in captured.out
        assert "Failed" in captured.out
        assert "00:02" in captured.out

    def test_setup_logger(self, training_manager, tmpdir):
        """Test _setup_logger method."""
        results_dir = str(tmpdir.mkdir("results"))
        logger = training_manager._setup_logger(results_dir)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "TrainingManager"
        assert len(logger.handlers) == 2  # File handler and console handler

        log_file = os.path.join(results_dir, "error_log.txt")
        assert os.path.exists(log_file)

    def test_run_experiments_successful(self, training_manager, mock_save_report, tmpdir):
        """Test the run_experiments method with a successful workflow execution."""
        results_dir = tmpdir.mkdir("results").mkdir("test_results")
        
        # Mock a concrete subclass of Workflow
        mock_workflow_class = create_autospec(Workflow, instance=False)
        mock_workflow_instance = MagicMock()
        mock_workflow_class.return_value = mock_workflow_instance  
        mock_workflow_instance.workflow = MagicMock()
        mock_workflow_class.__name__ = "MockWorkflow"

        with patch("brisk.training.TrainingManager.TrainingManager._get_results_dir", return_value=str(results_dir)), \
            patch("tqdm.tqdm") as tqdm_mock:

            training_manager.run_experiments(
                workflow=mock_workflow_class,
                create_report=False
            )
            
            # Verify workflow is called
            assert mock_workflow_instance.workflow.called, "Expected the workflow method to be called on the workflow instance."
        
            # Check that report is not created
            mock_save_report.assert_not_called()

            config_log_path = os.path.join(results_dir, "config_log.txt")
            assert os.path.exists(config_log_path)

    def test_run_experiments_with_existing_directory(self, training_manager):
        """Test run_experiments raises FileExistsError if results directory already exists."""
        results_name = "existing_results"
        
        # Mock the Workflow class and instance
        mock_workflow_class = create_autospec(Workflow, instance=False)
        mock_workflow_instance = MagicMock()
        mock_workflow_class.return_value = mock_workflow_instance
        mock_workflow_class.__name__ = "MockWorkflow"

        # Patch `os.path.exists` to simulate an existing directory
        with patch("os.path.exists", return_value=True), \
            pytest.raises(FileExistsError):
            training_manager.run_experiments(
                workflow=mock_workflow_class,
                results_name=results_name,
                create_report=True
            )

    def test_run_experiments_failed_workflow(self, training_manager, caplog, tmpdir):
        """Test run_experiments handles exceptions in workflow execution."""
        results_dir = tmpdir.mkdir("results").mkdir("test_results")

        # Mock the Workflow class and instance with a side effect to simulate an error
        mock_workflow_class = create_autospec(Workflow, instance=False)
        mock_workflow_instance = MagicMock()
        mock_workflow_instance.workflow.side_effect = Exception("Test Exception")
        mock_workflow_class.return_value = mock_workflow_instance
        mock_workflow_class.__name__ = "MockWorkflow"
        
        with patch("brisk.training.TrainingManager.TrainingManager._get_results_dir", return_value=str(results_dir)), \
            patch("tqdm.tqdm") as tqdm_mock:
            training_manager.run_experiments(workflow=mock_workflow_class, create_report=False)
            
            # Check that exception is logged
            assert "Test Exception" in caplog.text

    def test_run_experiments_creates_config_log(self, training_manager, tmpdir):
        """Test that configuration log is created."""
        # Set up the temporary results directory
        results_dir = tmpdir.mkdir("test_results")
        
        # Mock the Workflow class and instance
        mock_workflow_class = create_autospec(Workflow, instance=False)
        mock_workflow_instance = MagicMock()
        mock_workflow_class.return_value = mock_workflow_instance
        mock_workflow_class.__name__ = "MockWorkflow"

        with patch("os.makedirs"), \
            patch("tqdm.tqdm"), \
            patch("brisk.training.TrainingManager.TrainingManager._get_results_dir", return_value=str(results_dir)):
            training_manager.run_experiments(workflow=mock_workflow_class, create_report=False)
            
            # Verify the config log file was created
            config_log_path = os.path.join(results_dir, "config_log.txt")
            assert os.path.exists(config_log_path)

    def test_run_experiments_report_creation(self, training_manager, tmpdir):
        """Test run_experiments triggers report creation if create_report=True."""
        results_dir = tmpdir.mkdir("results").mkdir("test_results")

        # Mock the Workflow class and instance
        mock_workflow_class = create_autospec(Workflow, instance=False)
        mock_workflow_instance = MagicMock()
        mock_workflow_class.return_value = mock_workflow_instance
        mock_workflow_class.__name__ = "MockWorkflow"

        with patch("brisk.training.TrainingManager.ReportManager") as MockReportManager, \
            patch("brisk.training.TrainingManager.TrainingManager._get_results_dir", return_value=str(results_dir)), \
            patch("tqdm.tqdm") as tqdm_mock:
            
            mock_report_instance = MockReportManager.return_value
            training_manager.run_experiments(workflow=mock_workflow_class, create_report=True)
            
            # Verify that create_report was called on the ReportManager instance
            mock_report_instance.create_report.assert_called_once()
