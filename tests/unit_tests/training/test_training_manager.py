"""Unit tests for the TrainingManager class."""

# pylint: disable=protected-access, redefined-outer-name, inconsistent-quotes
# NOTE inconsistent quotes are due to the use of single quotes in some f-strings
# Double quotes can be used in Python 3.12, but not in Python 3.11 or 3.10.

import collections
import importlib
import os
import sys
import warnings
import pathlib

import pytest
from unittest import mock
from sklearn import linear_model

from brisk.data.data_manager import DataManager
from brisk.configuration.algorithm_wrapper import AlgorithmWrapper
from brisk.evaluation.metric_manager import MetricManager
from brisk.training.training_manager import TrainingManager
from brisk.configuration.experiment import Experiment
from brisk.training.workflow import Workflow
from brisk.version import __version__
from brisk.services.bundle import ServiceBundle

@pytest.fixture
def metric_config(mock_brisk_project):
    metric_file = mock_brisk_project / "metrics.py"
    spec = importlib.util.spec_from_file_location("metrics", str(metric_file))
    metric_module = importlib.util.module_from_spec(spec)
    sys.modules["metrics"] = metric_module
    spec.loader.exec_module(metric_module)

    return metric_module.METRIC_CONFIG


@pytest.fixture
def configuration(mock_brisk_project):
    """
    Create a configuration instance using the create_configuration function.
    """
    settings_file = mock_brisk_project / "settings.py"
    spec = importlib.util.spec_from_file_location(
        "settings", str(settings_file)
    )
    settings_module = importlib.util.module_from_spec(spec)
    sys.modules["settings"] = settings_module
    spec.loader.exec_module(settings_module)

    return settings_module.create_configuration()


@pytest.fixture
def workflow(mock_brisk_project):
    workflow_file = mock_brisk_project / "workflows" / "regression_workflow.py"
    spec = importlib.util.spec_from_file_location(
        "regression_workflow", str(workflow_file)
    )
    workflow_module = importlib.util.module_from_spec(spec)
    sys.modules["regression_workflow"] = workflow_module
    spec.loader.exec_module(workflow_module)

    return workflow_module.Regression


@pytest.fixture
def training_manager(metric_config, configuration):
    return TrainingManager(metric_config, configuration)


@pytest.fixture
def experiment():
    return Experiment(
        group_name="test_experiment",
        dataset_path="./datasets/regression.csv",
        algorithms={"model": AlgorithmWrapper(
            name="linear",
            display_name="Linear Regression",
            algorithm_class=linear_model.LinearRegression
        )},
        workflow="regression_workflow",
        workflow_args={},           
        table_name=None,
        categorical_features=None,
        split_index=0
    )


class TestTrainingManager:
    """Unit tests for the TrainingManager class."""
    def test_init(self, training_manager):
        assert isinstance(training_manager.services, ServiceBundle)

        assert isinstance(training_manager.results_dir, pathlib.Path)

        assert isinstance(training_manager.metric_config, MetricManager)

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

    @mock.patch("brisk.services.reporting.ReportingService.add_experiment_groups")
    @mock.patch("brisk.training.training_manager.TrainingManager._run_single_experiment")
    def test_run_experiments_resets_results(self, mock_run_single_experiment, mock_add_experiment_groups, training_manager, workflow):
        initial_experiment_results = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        initial_experiment_results["group1"]["exp1"] = ["experiment 1", 10]
        initial_experiment_results["group1"]["exp2"] = ["experiment 2", 12]
        initial_experiment_results["group2"]["exp1"] = ["experiment 3", 14]

        training_manager.experiment_results = initial_experiment_results
        # NOTE mock the _run_single_experiment method so no results are added
        # Set up workflow mapping for the test
        training_manager.workflow_mapping = {"regression_workflow": workflow}
        training_manager.run_experiments(False)

        assert training_manager.experiment_results == collections.defaultdict(
            lambda: collections.defaultdict(list)
        )

    @mock.patch("tqdm.tqdm")
    @mock.patch("brisk.services.reporting.ReportingService._collect_best_score")
    @mock.patch("brisk.services.reporting.ReportingService.add_experiment_groups")
    def  test_run_experiments_progress_bar(self, mock_add_experiment_groups, mock_collect_best_score, mock_tqdm, training_manager, workflow):
        mock_progress_bar = mock.Mock()
        mock_tqdm.return_value = mock_progress_bar
        
        # Add extra experiments to the queue
        training_manager.experiments.append(Experiment(
            group_name="group1",
            dataset_path="./datasets/regression.csv",
            algorithms={"model": AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression
            )},
            workflow="regression_workflow",
            workflow_args={},           
            table_name=None,
            categorical_features=None,
            split_index=0
        ))
        training_manager.experiments.append(Experiment(
            group_name="group2",
            dataset_path="./datasets/regression.csv",
            algorithms={"model": AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression
            )},
            workflow="regression_workflow",
            workflow_args={},           
            table_name=None,
            categorical_features=None,
            split_index=0
        ))
        
        # Set up workflow mapping for the test
        training_manager.workflow_mapping = {"regression_workflow": workflow}
        training_manager.run_experiments(False)
        
        # Verify progress bar was created with correct total
        mock_tqdm.assert_called_once_with(
            total=4,
            desc="Running Experiments",
            unit="experiment"
        )
        
        # Verify progress bar update was called for each experiment
        assert mock_progress_bar.update.call_count == 4
        for call in mock_progress_bar.update.call_args_list:
            args, _ = call
            assert args == (1,), f"Expected update(1), got update{args}"

    @mock.patch("brisk.training.training_manager.TrainingManager._run_single_experiment")
    @mock.patch("tqdm.tqdm")
    @mock.patch("brisk.services.reporting.ReportingService.add_experiment_groups")
    def test_run_experiments_consumes_experiments(self, mock_add_experiment_groups, mock_tqdm, mock_run_single_experiment, training_manager, workflow):
        mock_progress_bar = mock.Mock()
        mock_tqdm.return_value = mock_progress_bar
        
        initial_experiment_count = len(training_manager.experiments)
        initial_experiments = list(training_manager.experiments)
        assert initial_experiment_count > 0, "Training manager should have experiments for this test"
        
        # Set up workflow mapping for the test
        training_manager.workflow_mapping = {"regression_workflow": workflow}
        training_manager.run_experiments(False)
        
        # Verify experiments queue is empty after running
        assert len(training_manager.experiments) == 0, "Experiments queue should be empty after running"
        # Verify _run_single_experiment was called for each original experiment
        assert mock_run_single_experiment.call_count == initial_experiment_count
        # Verify each experiment was passed to _run_single_experiment
        called_experiments = [call[0][0] for call in mock_run_single_experiment.call_args_list]
        for original_exp, called_exp in zip(initial_experiments, called_experiments):
            assert original_exp == called_exp, "Experiments should be processed in order"
 
    @mock.patch("brisk.training.training_manager.TrainingManager._setup_workflow")
    @mock.patch("brisk.training.training_manager.TrainingManager._handle_success")
    @mock.patch("brisk.services.reporting.ReportingService.add_experiment_groups")
    def test_run_single_experiment_success(self, mock_add_experiment_groups, mock_setup_workflow, mock_handle_success, training_manager, experiment, workflow):
        # Set up workflow mapping for the test
        training_manager.workflow_mapping = {"regression_workflow": workflow}
        training_manager._run_single_experiment(experiment, "test_results")
        assert mock_setup_workflow.call_count == 1
        assert mock_handle_success.call_count == 1

    @mock.patch("brisk.training.training_manager.TrainingManager._handle_success")
    @mock.patch("brisk.training.training_manager.TrainingManager._handle_failure")
    @mock.patch("brisk.training.training_manager.TrainingManager._setup_workflow")
    @mock.patch("tqdm.tqdm.write")
    @mock.patch("time.time")
    @mock.patch("brisk.services.reporting.ReportingService.add_experiment")
    @pytest.mark.parametrize("error_type, error_message", [
        (ValueError, "Invalid value provided"),
        (TypeError, "Wrong type provided"),
        (AttributeError, "Attribute not found"),
        (KeyError, "Key not found"),
        (FileNotFoundError, "File not found"),
        (ImportError, "Cannot import module"),
        (MemoryError, "Out of memory"),
        (RuntimeError, "Runtime error occurred")
    ])
    def test_run_single_experiment_failure(
        self,
        mock_add_experiment,
        mock_time,
        mock_tqdm_write,
        mock_setup_workflow,
        mock_handle_failure,
        mock_handle_success,
        training_manager,
        workflow,
        error_type,
        error_message
    ):
        mock_time.return_value = 1234567890.0
        
        mock_experiment = mock.Mock()
        mock_experiment.group_name = "test_group"
        mock_experiment.dataset_name = "test_dataset"
        mock_experiment.name = "test_experiment"
        mock_experiment.workflow = "regression_workflow"
        
        # Create a mock workflow instance that raises the specified error
        mock_workflow_instance = mock.Mock()
        mock_workflow_instance.workflow.side_effect = error_type(error_message)
        mock_setup_workflow.return_value = mock_workflow_instance
        
        # Set up workflow mapping for the test
        training_manager.workflow_mapping = {"regression_workflow": workflow}
        training_manager._run_single_experiment(
            mock_experiment,
            "test_results_dir"
        )
        
        mock_setup_workflow.assert_called_once()        
        mock_workflow_instance.workflow.assert_called_once()        
        mock_handle_failure.assert_called_once_with(
            "test_group",           # group_name
            "test_dataset",         # dataset_name  
            "test_experiment",      # experiment_name
            1234567890.0,          # start_time
            mock_workflow_instance.workflow.side_effect  # the error instance
        )
        
        mock_handle_success.assert_not_called()
        
        # Verify the error passed to _handle_failure is the correct type and message
        call_args = mock_handle_failure.call_args[0]
        error_passed = call_args[4]  # 5th argument is the error
        assert isinstance(error_passed, error_type)
        assert str(error_passed).replace("'", "") == error_message

    @mock.patch("brisk.training.training_manager.TrainingManager._handle_success")
    @mock.patch("brisk.training.training_manager.TrainingManager._setup_workflow")
    @mock.patch("tqdm.tqdm.write")
    @mock.patch("brisk.services.reporting.ReportingService.add_experiment")
    def test_run_single_experiment_logging(
        self,
        mock_add_experiment,
        mock_tqdm_write,
        mock_setup_workflow,
        mock_handle_success,
        training_manager,
        workflow
    ):
        mock_experiment = mock.Mock()
        mock_experiment.group_name = "test_group"
        mock_experiment.dataset_name = "test_dataset"
        mock_experiment.name = "test_experiment"
        mock_experiment.workflow = "regression_workflow"
        
        # Create a mock workflow instance that succeeds
        mock_workflow_instance = mock.Mock()
        mock_setup_workflow.return_value = mock_workflow_instance
        
        # Set up workflow mapping for the test
        training_manager.workflow_mapping = {"regression_workflow": workflow}
        training_manager._run_single_experiment(
            mock_experiment,
            "test_results_dir"
        )
        
        expected_calls = [
            mock.call(f"\n{'=' * 80}"),
            mock.call(
                f"\nStarting experiment 'test_experiment' on dataset "
                f"'test_dataset' using workflow 'Regression'."
            )
        ]
        mock_tqdm_write.assert_has_calls(expected_calls, any_order=False)
        mock_workflow_instance.workflow.assert_called_once()        
        mock_handle_success.assert_called_once()

    @mock.patch("brisk.training.training_manager.TrainingManager._log_warning")
    @mock.patch("brisk.training.training_manager.TrainingManager._handle_success")
    @mock.patch("brisk.training.training_manager.TrainingManager._setup_workflow")
    @mock.patch("tqdm.tqdm.write")
    @mock.patch("warnings.warn")
    @mock.patch("brisk.services.reporting.ReportingService.add_experiment")
    def test_run_single_experiment_warning(
        self,
        mock_add_experiment,
        mock_warnings_warn,
        mock_tqdm_write,
        mock_setup_workflow,
        mock_handle_success,
        mock_log_warning,
        training_manager,
        workflow
    ):
        mock_experiment = mock.Mock()
        mock_experiment.group_name = "test_group" 
        mock_experiment.dataset_name = "test_dataset"
        mock_experiment.name = "test_experiment"
        mock_experiment.workflow = "regression_workflow"

        # Create a mock workflow instance that triggers a warning
        def trigger_warning():
            import warnings
            warnings.warn("Test warning message", UserWarning)

        mock_workflow_instance = mock.Mock()
        mock_workflow_instance.workflow = trigger_warning
        mock_setup_workflow.return_value = mock_workflow_instance
        
        original_showwarning = warnings.showwarning
        
        try:
            # Set up workflow mapping for the test
            training_manager.workflow_mapping = {"regression_workflow": workflow}
            training_manager._run_single_experiment(
                mock_experiment,
                "test_results_dir"
            )
            
            # Trigger the warning manually to test the custom handler
            # Since we can't easily capture the lambda function that gets assigned,
            # we'll verify the behavior by checking that warnings.showwarning was modified
            assert warnings.showwarning != original_showwarning, \
                "warnings.showwarning should be modified during experiment execution"
            
            # Test the custom warning handler directly
            test_message = "Test warning message"
            test_category = UserWarning
            test_filename = "test_file.py"
            test_lineno = 123
            
            warnings.showwarning(
                test_message,
                test_category,
                test_filename,
                test_lineno
            )
            
            mock_log_warning.assert_called_with(
                test_message,
                test_category,
                test_filename,
                test_lineno,
                "test_dataset",
                "test_experiment"
            )
            
        finally:
            warnings.showwarning = original_showwarning
        
        mock_handle_success.assert_called_once()

    def test_reset_experiment_results(self, training_manager):
        training_manager._reset_experiment_results()
        assert training_manager.experiment_results == collections.defaultdict(
            lambda: collections.defaultdict(list)
        )

    @mock.patch("brisk.reporting.report_renderer.ReportRenderer.render")
    def test_create_report(self, mock_create_report, training_manager):
        training_manager._create_report("test_results_dir")
        mock_create_report.assert_called_once()



    def test_setup_workflow(self, workflow, training_manager):
        training_manager.logger = mock.Mock()

        group_name = "test_group"
        current_experiment = Experiment(
            group_name=group_name,
            dataset_path="./datasets/regression.csv",
            algorithms={"model": AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression
            )},
            workflow="regression_workflow",
            workflow_args={},           
            table_name=None,
            categorical_features=None,
            split_index=0
        )
        results_dir = ""
        dataset_name = "regression"
        experiment_name = current_experiment.name
        expected_algo_names = ["linear"]
        split = "split_0"

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
                [split]
                [experiment_name]
                ) == os.path.join(
                    results_dir, group_name, dataset_name, split, experiment_name
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

        # training_manager.logger.exception.assert_called_once_with(
        #     expected_error_message
        # )
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

    def test_get_experiment_dir(self, tmp_path, training_manager):
        """Test creating a new experiment directory."""
        results_dir = str(tmp_path / "test_results")
        group_name = "test_group"
        dataset_name = "test_dataset"
        experiment_name = "test_experiment"
        split = 0
        
        result_path = training_manager._get_experiment_dir(
            results_dir, group_name, dataset_name, split, experiment_name
        )
        
        expected_path = os.path.normpath(
            os.path.join(results_dir, group_name, dataset_name, "split_0", experiment_name)
        )
        assert result_path == expected_path
        assert os.path.exists(result_path)
        assert os.path.isdir(result_path)
        assert os.path.exists(os.path.join(results_dir, group_name))
        assert os.path.exists(os.path.join(results_dir, group_name, dataset_name))

    def test_get_experiment_dir_exists(self, tmp_path, training_manager):
        """Test that method works correctly when directory already exists."""
        results_dir = str(tmp_path / "test_results")
        group_name = "test_group"
        dataset_name = "test_dataset"
        experiment_name = "test_experiment"
        
        expected_path = os.path.normpath(
            os.path.join(results_dir, group_name, dataset_name, "split_0", experiment_name)
        )
        os.makedirs(expected_path, exist_ok=True)
        
        # Verify it exists before calling the method
        assert os.path.exists(expected_path)
        
        result_path = training_manager._get_experiment_dir(
            results_dir, group_name, dataset_name, 0, experiment_name
        )        
        assert result_path == expected_path
        assert os.path.exists(result_path)
        assert os.path.isdir(result_path)

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
    def test_cleanup_empty_error_log(
        self,
        mock_logging_shutdown,
        tmp_path,
        training_manager,
    ):
        results_dir = str(tmp_path)
        progress_bar = mock.MagicMock()

        error_log_path = tmp_path / "error_log.txt"
        error_log_path.write_text("")  # Empty file

        # Verify file exists and is empty before cleanup
        assert error_log_path.exists()
        assert error_log_path.stat().st_size == 0
        
        training_manager._cleanup(results_dir, progress_bar)        
        progress_bar.close.assert_called_once()
        mock_logging_shutdown.assert_called_once()    
        assert not error_log_path.exists()

    @mock.patch("brisk.training.training_manager.logging.shutdown")
    def test_cleanup_non_empty_error_log(
        self,
        mock_logging_shutdown,
        tmp_path,
        training_manager,
    ):
        results_dir = str(tmp_path)
        progress_bar = mock.MagicMock()

        error_log_path = tmp_path / "error_log.txt"
        error_log_path.write_text("Some error occurred\nAnother error\n")

        # Verify file exists and has content before cleanup
        assert error_log_path.exists()
        assert error_log_path.stat().st_size > 0

        training_manager._cleanup(results_dir, progress_bar)
        progress_bar.close.assert_called_once()
        mock_logging_shutdown.assert_called_once()        
        assert error_log_path.exists()
        assert error_log_path.stat().st_size > 0

    @mock.patch("brisk.training.training_manager.logging.shutdown")
    def test_cleanup_no_error_log(
        self,
        mock_logging_shutdown,
        tmp_path,
        training_manager,
    ):
        results_dir = str(tmp_path)
        progress_bar = mock.MagicMock()
        
        # Don't create any error log file
        error_log_path = tmp_path / "error_log.txt"
        assert not error_log_path.exists()
        
        training_manager._cleanup(results_dir, progress_bar)        
        progress_bar.close.assert_called_once()
        mock_logging_shutdown.assert_called_once()
        assert not error_log_path.exists()
