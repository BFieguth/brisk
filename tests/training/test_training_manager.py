import os
from unittest import mock
from collections import deque
from datetime import datetime

import pytest

from brisk.training.TrainingManager import TrainingManager
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper

@pytest.fixture
def setup_training_manager():
    """
    Setup for the TrainingManager instance.
    Mocks required objects for unit testing.
    """
    model_mock = mock.MagicMock()
    
    wrapper_mock = mock.MagicMock(spec=AlgorithmWrapper)
    wrapper_mock.instantiate.return_value = model_mock 

    method_config = {
        'linear': wrapper_mock 
    }
    
    metric_config = mock.MagicMock()
    splitter = mock.MagicMock()
    workflow = mock.MagicMock()

    splitter.split.return_value = (
        mock.MagicMock(),  
        mock.MagicMock(),  
        mock.MagicMock(),  
        mock.MagicMock()   
    )

    methods = ['linear']
    data_paths = [('dataset.csv', None)]
    
    return TrainingManager(
        method_config, metric_config, workflow, splitter, methods, data_paths
    )


class TestTrainingManager:
    @mock.patch("os.makedirs")
    def test_results_directory_with_timestamp(self, mock_makedirs, setup_training_manager):
        """
        Test that a new directory with a timestamp is created when running experiments.
        """
        tm = setup_training_manager

        mock_now = datetime(2023, 9, 6, 15, 30, 0)
        with mock.patch("brisk.training.TrainingManager.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now
            mock_datetime.strftime = datetime.strftime

            print(f"Running configurations with setup: {tm.results_dir}")
            tm.run_experiments(create_report=False)

            expected_dir = "06_09_2023_15_30_00_Results/linear_dataset"
            print(f"Expected directory to be created: {expected_dir}")
            mock_makedirs.assert_called_with(expected_dir)

    @mock.patch("brisk.training.TrainingManager.os.makedirs")
    def test_user_defined_results_dir(self, mock_makedirs, setup_training_manager):
        """
        Test that the user-defined results directory is used when provided.
        """
        tm = TrainingManager(
            method_config=setup_training_manager.method_config,
            metric_config=setup_training_manager.metric_config,
            workflow=setup_training_manager.workflow,
            splitter=setup_training_manager.splitter,
            methods=setup_training_manager.methods,
            data_paths=setup_training_manager.data_paths,
            results_dir="user_results_dir"
        )

        tm.run_experiments(create_report=False)
        mock_makedirs.assert_called_with("user_results_dir/linear_dataset")

    def test_consume_experiments(self, setup_training_manager):
        """
        Test that experiments are consumed as they are processed.
        """
        tm = setup_training_manager
        assert len(tm.experiments) == 1

        tm.run_experiments(create_report=False)
        assert len(tm.experiments) == 0

    @mock.patch("os.makedirs")
    def test_run_experiments_creates_correct_directory_structure(self, mock_makedirs, setup_training_manager):
        """
        Test that the correct directory structure is created for each experiment.
        """
        results_dir = "test_results_dir"
        tm = TrainingManager(
            method_config=setup_training_manager.method_config,
            metric_config=setup_training_manager.metric_config,
            workflow=setup_training_manager.workflow,
            splitter=setup_training_manager.splitter,
            methods=setup_training_manager.methods,
            data_paths=setup_training_manager.data_paths,
            results_dir=results_dir 
        )
        mock_workflow_function = mock.MagicMock()

        tm.run_experiments(create_report=False)

        expected_dir = os.path.join(tm.results_dir, "linear_dataset")
        mock_makedirs.assert_any_call(expected_dir)

    @mock.patch("os.makedirs")
    @mock.patch("brisk.Workflow.workflow")
    def test_run_experiments_calls_workflow_function(self, mock_makedirs, mock_workflow_method, setup_training_manager):
        """
        Test that the user-defined workflow function is called correctly for each experiment.
        """
        tm = setup_training_manager
        tm.run_experiments(create_report=False)

        assert mock_workflow_method.call_count == 1

    def test_invalid_method_in_experiment(self, setup_training_manager):
        """
        Test that ValueError is raised when a method not in the method_config is used.
        """
        invalid_methods = ['not_a_model']
        
        with pytest.raises(ValueError, match="The following methods are not included in the configuration: \['not_a_model'\]"):
            tm = TrainingManager(
                method_config=setup_training_manager.method_config,
                metric_config=setup_training_manager.metric_config,
                workflow=setup_training_manager.workflow,
                splitter=setup_training_manager.splitter,
                methods=invalid_methods,
                data_paths=setup_training_manager.data_paths,
            )

    @mock.patch("os.path.exists")
    @mock.patch("os.makedirs")
    def test_results_directory_already_exists(self, mock_makedirs, mock_exists, setup_training_manager):
        """
        Test that FileExistsError is raised when the results directory already exists.
        """
        mock_exists.return_value = True
        
        with pytest.raises(FileExistsError, match="Results directory 'existing_dir' already exists."):
            TrainingManager(
                method_config=setup_training_manager.method_config,
                metric_config=setup_training_manager.metric_config,
                workflow=setup_training_manager.workflow,
                splitter=setup_training_manager.splitter,
                methods=setup_training_manager.methods,
                data_paths=setup_training_manager.data_paths,
                results_dir="existing_dir"
            )
            