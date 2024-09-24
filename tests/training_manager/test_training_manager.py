import os
from unittest import mock
from collections import deque
from datetime import datetime

import pytest

from ml_toolkit.training.TrainingManager import TrainingManager
from ml_toolkit.utility.ModelWrapper import ModelWrapper

@pytest.fixture
def setup_training_manager():
    """
    Setup for the TrainingManager instance.
    Mocks required objects for unit testing.
    """
    model_mock = mock.MagicMock()
    
    wrapper_mock = mock.MagicMock(spec=ModelWrapper)
    wrapper_mock.instantiate.return_value = model_mock 

    method_config = {
        'linear': wrapper_mock 
    }
    
    scoring_config = mock.MagicMock()
    splitter = mock.MagicMock()

    splitter.split.return_value = (
        mock.MagicMock(),  
        mock.MagicMock(),  
        mock.MagicMock(),  
        mock.MagicMock()   
    )

    methods = ['linear']
    data_paths = [('dataset.csv', None)]
    
    return TrainingManager(
        method_config, scoring_config, splitter, methods, data_paths
    )


class TestTrainingManager:
    @mock.patch("os.makedirs")
    def test_results_directory_with_timestamp(self, mock_makedirs, setup_training_manager):
        """
        Test that a new directory with a timestamp is created when running experiments.
        """
        tm = setup_training_manager

        mock_now = datetime(2023, 9, 6, 15, 30, 0)
        with mock.patch("ml_toolkit.training.TrainingManager.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now
            mock_datetime.strftime = datetime.strftime

            print(f"Running configurations with setup: {tm.results_dir}")
            tm.run_experiments(mock.MagicMock(), create_report=False)

            expected_dir = "06_09_2023_15_30_00_Results/linear_dataset"
            print(f"Expected directory to be created: {expected_dir}")
            mock_makedirs.assert_called_with(expected_dir)

    @mock.patch("ml_toolkit.training.TrainingManager.os.makedirs")
    def test_user_defined_results_dir(self, mock_makedirs, setup_training_manager):
        """
        Test that the user-defined results directory is used when provided.
        """
        tm = TrainingManager(
            method_config=setup_training_manager.method_config,
            scoring_config=setup_training_manager.scoring_config,
            splitter=setup_training_manager.splitter,
            methods=setup_training_manager.methods,
            data_paths=setup_training_manager.data_paths,
            results_dir="user_results_dir"
        )

        tm.run_experiments(mock.MagicMock(), create_report=False)
        mock_makedirs.assert_called_with("user_results_dir/linear_dataset")

    def test_consume_experiments(self, setup_training_manager):
        """
        Test that experiments are consumed as they are processed.
        """
        tm = setup_training_manager
        assert len(tm.experiments) == 1

        tm.run_experiments(mock.MagicMock(), create_report=False)
        assert len(tm.experiments) == 0

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("os.makedirs")
    def test_error_logging(self, mock_makedirs, mock_open, setup_training_manager):
        """
        Test that errors are correctly logged when a experiment fails.
        """
        results_dir = "test_results_dir"
        tm = TrainingManager(
            method_config=setup_training_manager.method_config,
            scoring_config=setup_training_manager.scoring_config,
            splitter=setup_training_manager.splitter,
            methods=setup_training_manager.methods,
            data_paths=setup_training_manager.data_paths,
            results_dir=results_dir 
        )
        mock_workflow_function = mock.MagicMock(side_effect=Exception("Test Error"))

        tm.run_experiments(mock_workflow_function, create_report=False)

        mock_open.assert_called_with(os.path.join(tm.results_dir, "error_log.txt"), "w")
        handle = mock_open()
        handle.write.assert_called_with("Error for ('linear',) on ('dataset.csv', None): Test Error")

    @mock.patch("os.makedirs")
    def test_run_experiments_creates_correct_directory_structure(self, mock_makedirs, setup_training_manager):
        """
        Test that the correct directory structure is created for each experiment.
        """
        results_dir = "test_results_dir"
        tm = TrainingManager(
            method_config=setup_training_manager.method_config,
            scoring_config=setup_training_manager.scoring_config,
            splitter=setup_training_manager.splitter,
            methods=setup_training_manager.methods,
            data_paths=setup_training_manager.data_paths,
            results_dir=results_dir 
        )
        mock_workflow_function = mock.MagicMock()

        tm.run_experiments(mock_workflow_function, create_report=False)

        expected_dir = os.path.join(tm.results_dir, "linear_dataset")
        mock_makedirs.assert_any_call(expected_dir)

    @mock.patch("os.makedirs")
    def test_run_experiments_calls_workflow_function(self, mock_makedirs, setup_training_manager):
        """
        Test that the user-defined workflow function is called correctly for each experiment.
        """
        tm = setup_training_manager

        mock_workflow_function = mock.MagicMock()

        tm.run_experiments(mock_workflow_function, create_report=False)

        assert mock_workflow_function.call_count == 1

    def test_invalid_method_in_experiment(self, setup_training_manager):
        """
        Test that ValueError is raised when a method not in the method_config is used.
        """
        invalid_methods = ['not_a_model']
        
        with pytest.raises(ValueError, match="The following methods are not included in the configuration: \['not_a_model'\]"):
            tm = TrainingManager(
                method_config=setup_training_manager.method_config,
                scoring_config=setup_training_manager.scoring_config,
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
                scoring_config=setup_training_manager.scoring_config,
                splitter=setup_training_manager.splitter,
                methods=setup_training_manager.methods,
                data_paths=setup_training_manager.data_paths,
                results_dir="existing_dir"
            )
            