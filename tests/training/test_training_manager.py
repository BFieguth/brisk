import datetime
import os
from unittest.mock import patch, MagicMock, Mock, create_autospec
from collections import deque
import logging
import warnings

import pandas as pd
import pytest
from pathlib import Path

from brisk.training.training_manager import TrainingManager
from brisk.utility.algorithm_wrapper import AlgorithmWrapper
from brisk.data.data_split_info import DataSplitInfo
from brisk.training.workflow import Workflow
from brisk.configuration.experiment import Experiment

class MockDataManager:
    def split(self, data_path, group_name, filename):
        split_info = DataSplitInfo(
            X_train=pd.DataFrame([[1, 2], [3, 4]]),
            X_test=pd.DataFrame([[5, 6]]),
            y_train=pd.Series([0, 1]),
            y_test=pd.Series([1]),
            filename=filename,
            scaler=None,
            features=['feature1', 'feature2'],
            categorical_features=['feature1']
        )
        return split_info


@pytest.fixture
def metric_config():
    return {
        'mean_squared_error': {
            'display_name': 'Mean Squared Error',
            'func': MagicMock(),
            'scorer': MagicMock(),
        },
        'mean_absolute_error': {
            'display_name': 'Mean Absolute Error',
            'func': MagicMock(),
            'scorer': MagicMock(),
        }
    }


@pytest.fixture
def data_managers():
    return {
        'group1': MockDataManager(),
        'group2': MockDataManager()
    }


@pytest.fixture
def experiments():
    linear = AlgorithmWrapper(
        name="linear",
        display_name="Linear Regression",
        algorithm_class=Mock()
    )
    ridge = AlgorithmWrapper(
        name="ridge",
        display_name="Ridge Regression",
        algorithm_class=Mock()
    )
    
    exp1 = Experiment(
        group_name="group1",
        dataset=Path("data/test1.csv"),
        algorithms={"model": linear}
    )
    exp2 = Experiment(
        group_name="group2",
        dataset=Path("data/test2.csv"),
        algorithms={"model": ridge}
    )
    
    return deque([exp1, exp2])


@pytest.fixture
def output_structure():
    return {
        'group1': {
            'test1': ('data/test1.csv', 'group1')
        },
        'group2': {
            'test2': ('data/test2.csv', 'group2')
        }
    }


@pytest.fixture
def config_manager(data_managers, experiments, output_structure):
    mock_config_manager = MagicMock()    
    mock_config_manager.data_managers = data_managers
    mock_config_manager.experiments = experiments
    mock_config_manager.logfile = "# Test Config"
    mock_config_manager.output_structure = output_structure
    mock_config_manager.description_map = {}
    return mock_config_manager


@pytest.fixture
def training_manager(metric_config, config_manager):
    return TrainingManager(
        metric_config=metric_config,
        config_manager=config_manager,
        verbose=False
    )


class TestTrainingManager:
    def test_initialization(self, training_manager):
        """Test proper initialization of TrainingManager."""
        assert training_manager.metric_config is not None
        assert training_manager.data_managers is not None
        assert training_manager.logfile == "# Test Config"
        assert isinstance(training_manager.experiment_paths, dict)

    def test_get_results_dir(self, training_manager):
        """Test results directory creation with timestamp."""
        with patch('brisk.training.training_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "01_01_2024_00_00_00"
            result = training_manager._get_results_dir()
            assert result == os.path.join("results", "01_01_2024_00_00_00")

    def test_get_experiment_dir(self, training_manager, tmp_path):
        """Test experiment directory creation."""
        results_dir = str(tmp_path)
        exp_dir = training_manager._get_experiment_dir(
            results_dir=results_dir,
            group_name="group1",
            dataset_name="dataset1",
            experiment_name="experiment1"
        )
        expected_path = os.path.join(results_dir, "group1", "dataset1", "experiment1")
        assert exp_dir == expected_path
        assert os.path.exists(exp_dir)

    # TODO (Issue #72): Need to refactor to use the conftest fixtures
    # @patch('brisk.training.training_manager.report.ReportManager')
    # def test_run_experiments_success(self, mock_report_manager, training_manager, tmp_path):
    #     """Test successful experiment run."""
    #     # Mock workflow
    #     mock_workflow = Mock()
    #     mock_workflow.__name__ = "MockWorkflow"
    #     mock_workflow_instance = Mock()
    #     mock_workflow.return_value = mock_workflow_instance

    #     with patch('brisk.training.training_manager.os.makedirs'), \
    #          patch.object(training_manager, '_get_results_dir', return_value=str(tmp_path)), \
    #          patch.object(training_manager, '_save_config_log'), \
    #          patch.object(training_manager, '_save_data_distributions'), \
    #          patch.object(training_manager, '_setup_logger'):

    #         training_manager.run_experiments(
    #             workflow=mock_workflow,
    #             workflow_config={"param": "value"},
    #             create_report=True
    #         )

            # assert len(training_manager.experiment_paths) > 0
            # assert mock_workflow_instance.workflow.called
            # mock_report_manager.assert_called_once()

    # def test_run_experiments_failure(self, training_manager, tmp_path):
    #     """Test experiment failure handling."""
    #     mock_workflow = Mock()
    #     mock_workflow.__name__ = "MockWorkflow"
    #     mock_workflow_instance = Mock()
    #     mock_workflow_instance.workflow.side_effect = Exception("Test error")
    #     mock_workflow.return_value = mock_workflow_instance

    #     # Create a real logger for testing
    #     logger = logging.getLogger("TrainingManager")
    #     logger.setLevel(logging.DEBUG)
        
    #     # Create and add a file handler
    #     error_log_path = os.path.join(tmp_path, "error_log.txt")
    #     file_handler = logging.FileHandler(error_log_path)
    #     file_handler.setLevel(logging.WARNING)
    #     logger.addHandler(file_handler)

    #     with patch('brisk.training.training_manager.os.makedirs'), \
    #         patch.object(training_manager, '_get_results_dir', return_value=str(tmp_path)), \
    #         patch.object(training_manager, '_save_config_log'), \
    #         patch.object(training_manager, '_save_data_distributions'), \
    #         patch.object(training_manager, '_setup_logger', return_value=logger):

    #         training_manager.run_experiments(
    #             workflow=mock_workflow, create_report=False
    #         )
            
    #         logger.handlers = []
    #         file_handler.close()
            
    #         assert os.path.exists(error_log_path)
    #         with open(error_log_path, 'r') as f:
    #             content = f.read()
    #             assert "Test error" in content

    def test_save_data_distributions(self, training_manager, tmp_path):
        """Test saving data distributions."""
        with patch.object(training_manager.data_managers['group1'], 'split') as mock_split:
            mock_split_info = Mock(spec=DataSplitInfo)
            mock_split_info.save_distribution = Mock()
            mock_split.return_value = mock_split_info

            training_manager._save_data_distributions(
                str(tmp_path),
                {'group1': {'dataset1': ('path/to/data.csv', 'group1')}}
            )

            # Verify split_info.save_distribution was called
            mock_split_info.save_distribution.assert_called_once()

    def test_save_config_log(self, training_manager, tmp_path):
        """Test configuration log saving."""
        mock_workflow = Mock()
        mock_workflow.__name__ = "MockWorkflow"
        workflow_config = {"param": "value"}

        training_manager._save_config_log(
            str(tmp_path),
            mock_workflow,
            workflow_config,
            "# Test Config"
        )

        config_log_path = os.path.join(tmp_path, "config_log.md")
        assert os.path.exists(config_log_path)
        with open(config_log_path, 'r') as f:
            content = f.read()
            assert "MockWorkflow" in content
            assert "# Test Config" in content

    def test_setup_logger(self, training_manager, tmp_path):
        """Test logger setup."""
        logger = training_manager._setup_logger(str(tmp_path))
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "TrainingManager"
        assert len(logger.handlers) == 2  # File and console handlers
        
        # Test log levels
        file_handler = logger.handlers[0]
        console_handler = logger.handlers[1]
        assert file_handler.level == logging.WARNING
        assert console_handler.level == logging.ERROR  # Since verbose=False
