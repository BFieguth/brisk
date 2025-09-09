"""Training management system for machine learning experiments.

This module provides comprehensive functionality for managing the training and
evaluation of machine learning models across multiple datasets and algorithms.
The TrainingManager class coordinates the entire experiment lifecycle, from
setup through execution to reporting, ensuring robust handling of failures
and comprehensive result tracking.

The module integrates with the broader Brisk ecosystem, utilizing services
for logging, reporting, evaluation, and configuration management. It provides
a centralized way to orchestrate complex machine learning workflows while
maintaining detailed tracking and error handling.

Examples
--------
>>> from brisk.training.training_manager import TrainingManager
>>> from brisk.evaluation import metric_manager
>>> from brisk.configuration import configuration
>>> 
>>> # Create metric configuration
>>> metric_config = metric_manager.MetricManager()
>>> 
>>> # Create configuration manager
>>> config_manager = configuration.ConfigurationManager()
>>> 
>>> # Initialize training manager
>>> trainer = TrainingManager(metric_config, config_manager)
>>> 
>>> # Run all experiments
>>> trainer.run_experiments(create_report=True)
"""

import collections
import os
import time
import warnings
from typing import Optional, Type
from pathlib import Path

import tqdm

from brisk.evaluation import evaluation_manager, metric_manager
from brisk.reporting import report_renderer
from brisk.configuration import configuration_manager, experiment
from brisk.version import __version__
from brisk.training import workflow as workflow_module
from brisk.services import get_services

warnings.filterwarnings(
    "ignore",
    message="Filename: <_io.BytesIO object at.*>",
    category=UserWarning,
    module="plotnine"
)

class TrainingManager:
    """Manage the training and evaluation of machine learning models.
    
    This class coordinates the entire lifecycle of machine learning experiments,
    from setup through execution to reporting. It manages model training across
    multiple datasets and algorithms, ensuring robust error handling and
    comprehensive result tracking.
    
    The TrainingManager integrates with the broader Brisk ecosystem, utilizing
    services for logging, reporting, evaluation, and configuration management.
    It provides a centralized way to orchestrate complex machine learning
    workflows while maintaining detailed tracking and error handling.
    
    Parameters
    ----------
    metric_config : MetricManager
        Configuration for evaluation metrics and scoring
    config_manager : ConfigurationManager
        Instance containing all data and configuration needed to run experiments
        
    Attributes
    ----------
    services : ServiceBundle
        Bundle of all available services (logging, reporting, I/O, etc.)
    results_dir : str
        Directory where experiment results are stored
    metric_config : MetricManager
        Configuration for evaluation metrics and scoring
    eval_manager : EvaluationManager
        Manager for handling model evaluation and metrics
    data_managers : dict
        Maps group names to their corresponding data managers
    experiments : collections.deque
        Queue of experiments to run
    logfile : str
        Path to the configuration log file
    output_structure : dict
        Structure of output data organization
    description_map : dict
        Mapping of names to descriptions
    experiment_groups : dict
        Mapping of experiment group names to their configurations
    workflow_mapping : dict
        Maps experiment group names to their assigned workflow classes
    experiment_paths : defaultdict
        Nested structure tracking experiment output paths
    experiment_results : defaultdict
        Stores results of all experiments with status and timing
        
    Notes
    -----
    The TrainingManager uses a workflow-based approach where different
    experiment groups can use different workflow classes. This allows for
    flexibility in handling different types of machine learning tasks
    (classification, regression, etc.) with appropriate workflows.
    
    Error handling is comprehensive - individual experiment failures don't
    stop the overall process, and detailed logging is maintained for
    debugging and analysis.
    
    Examples
    --------
    >>> from brisk.training.training_manager import TrainingManager
    >>> from brisk.evaluation import metric_manager
    >>> from brisk.configuration import configuration
    >>> 
    >>> # Create metric configuration
    >>> metric_config = metric_manager.MetricManager()
    >>> 
    >>> # Create configuration manager with experiments
    >>> config_manager = configuration.ConfigurationManager()
    >>> 
    >>> # Initialize training manager
    >>> trainer = TrainingManager(metric_config, config_manager)
    >>> 
    >>> # Run all experiments with report generation
    >>> trainer.run_experiments(create_report=True)
    """
    def __init__(
        self,
        metric_config: metric_manager.MetricManager,
        config_manager: configuration_manager.ConfigurationManager
    ) -> None:
        """Initialize the TrainingManager with configuration and services.
        
        This constructor sets up the training manager with all necessary
        components for running machine learning experiments. It initializes
        services, evaluation managers, and data structures for tracking
        experiments and results.
        
        Parameters
        ----------
        metric_config : MetricManager
            Configuration for evaluation metrics and scoring
        config_manager : ConfigurationManager
            Instance containing all data and configuration needed to run
            experiments
            
        Notes
        -----
        The constructor initializes several key components:
        - Services bundle for logging, reporting, I/O, etc.
        - Evaluation manager for handling model evaluation
        - Data managers for different experiment groups
        - Experiment queue and workflow mappings
        - Result tracking structures
        
        The experiment_paths structure is a nested defaultdict that tracks
        output paths for each experiment group, dataset, split, and experiment.
        """
        self.services = get_services()
        self.results_dir = self.services.io.results_dir
        self.metric_config = metric_config
        self.eval_manager = evaluation_manager.EvaluationManager(
            self.metric_config
        )

        self.data_managers = config_manager.data_managers
        self.experiments = config_manager.experiment_queue
        self.logfile = config_manager.logfile
        self.output_structure = config_manager.output_structure
        self.description_map = config_manager.description_map
        self.experiment_groups = config_manager.experiment_groups
        self.workflow_mapping = config_manager.workflow_map
        self.experiment_paths = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: {}
                )
            )
        )
        self.experiment_results = None
        self._reset_experiment_results()

    def run_experiments(
        self,
        create_report: bool = True
    ) -> None:
        """Run all experiments in the queue and generate a comprehensive report.
        
        This method orchestrates the execution of all experiments in the queue,
        using the workflow_mapping to determine which workflow class to use for
        each experiment group. It provides comprehensive error handling and
        progress tracking throughout the process.
        
        The method ensures that all experiments are attempted even if some fail,
        and provides detailed logging and result tracking. After all experiments
        complete, it can generate an HTML report summarizing the results.
        
        Parameters
        ----------
        create_report : bool, default=True
            Whether to generate an HTML report after all experiments complete.
            If True, creates a comprehensive report with all results and
            visualizations.
            
        Raises
        ------
        ValueError
            If any experiment group does not have a workflow assigned in the
            workflow_mapping
            
        Notes
        -----
        The method performs the following steps:
        1. Resets experiment results tracking
        2. Creates a progress bar for monitoring
        3. Processes each experiment in the queue
        4. Handles individual experiment failures gracefully
        5. Prints a summary of all experiment results
        6. Exits early if all experiments failed
        7. Generates HTML report if requested
        8. Exports rerun configuration for reproducibility
        
        Examples
        --------
        >>> trainer = TrainingManager(metric_config, config_manager)
        >>> 
        >>> # Run experiments with report generation
        >>> trainer.run_experiments(create_report=True)
        >>> 
        >>> # Run experiments without report generation
        >>> trainer.run_experiments(create_report=False)
        """
        self._reset_experiment_results()
        progress_bar = tqdm.tqdm(
            total=len(self.experiments),
            desc="Running Experiments",
            unit="experiment"
        )

        while self.experiments:
            current_experiment = self.experiments.popleft()
            self._run_single_experiment(
                current_experiment,
                self.results_dir
            )
            progress_bar.update(1)

        self._print_experiment_summary()
        all_failed = all(
            result["status"] == "FAILED"
            for group_datasets in self.experiment_results.values()
            for experiments in group_datasets.values()
            for result in experiments
        )
        if all_failed:
            self._cleanup(self.results_dir, progress_bar)
            self.services.logger.logger.error(
                "All experiments failed. Exiting without creating a report."
            )
            return

        self.services.reporting.add_experiment_groups(self.experiment_groups)
        self._cleanup(self.results_dir, progress_bar)
        if create_report:
            self._create_report(self.results_dir)

        try:
            self.services.rerun.export_and_save(Path(self.results_dir))
        except (ValueError, TypeError, FileNotFoundError) as e:
            self.services.logger.logger.warning(
                f"Failed to save rerun config: {e}"
            )

    def _run_single_experiment(
        self,
        current_experiment: experiment.Experiment,
        results_dir: str
    ) -> None:
        """Run a single experiment and handle its outcome.
        
        This method sets up the experiment environment, determines the
        appropriate workflow from the workflow_mapping based on the experiment's
        group name, and executes the experiment with comprehensive error
        handling.
        
        The method handles both successful and failed experiments, logging
        appropriate information and updating the experiment results tracking.
        It also sets up warning handling to capture and log warnings during
        experiment execution.
        
        Parameters
        ----------
        current_experiment : Experiment
            The experiment to run, containing all necessary configuration
            and data paths
        results_dir : str
            Directory where experiment results will be stored
            
        Raises
        ------
        KeyError
            If the experiment's group name is not found in workflow_mapping
            
        Notes
        -----
        The method performs the following steps:
        1. Extracts experiment metadata (group, dataset, name, workflow)
        2. Sets up reporting context for the experiment
        3. Configures warning handling for detailed logging
        4. Attempts to set up and run the workflow
        5. Handles success or failure appropriately
        6. Updates experiment results tracking
        7. Clears reporting context
        
        Error handling covers a wide range of exceptions including ValueError,
        TypeError, AttributeError, KeyError, FileNotFoundError, ImportError,
        MemoryError, and RuntimeError.
        """
        success = False
        start_time = time.time()

        group_name = current_experiment.group_name
        dataset_name = current_experiment.dataset_name
        experiment_name = current_experiment.name
        workflow_class = self.workflow_mapping[current_experiment.workflow]

        self.services.reporting.set_context(
            group_name, dataset_name, current_experiment.split_index, None,
            current_experiment.algorithm_names
        )

        if dataset_name[1] is None:
            dataset = current_experiment.dataset_path.name
        else:
            dataset = dataset_name

        tqdm.tqdm.write(f"\n{'=' * 80}") # pylint: disable=W1405
        tqdm.tqdm.write(
            f"\nStarting experiment '{experiment_name}' on dataset "
            f"'{dataset}' (Split {current_experiment.split_index}) using "
            f"workflow '{workflow_class.__name__}'."
        )

        warnings.showwarning = (
            lambda message, category, filename, lineno, file=None, line=None: self._log_warning( # pylint: disable=line-too-long
                message,
                category,
                filename,
                lineno,
                dataset_name,
                experiment_name
            )
        )

        try:
            workflow_instance = self._setup_workflow(
                current_experiment, workflow_class, results_dir
            )
            workflow_instance.run()
            success = True

        except (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            FileNotFoundError,
            ImportError,
            MemoryError,
            RuntimeError
        ) as e:
            self._handle_failure(
                group_name,
                experiment_name,
                start_time,
                e,
                dataset,
                current_experiment.split_index
            )
            self.services.reporting.add_experiment(
                current_experiment.algorithms
            )
            self.services.reporting.clear_context()

        if success:
            self._handle_success(
                start_time,
                group_name,
                experiment_name,
                dataset,
                current_experiment.split_index
            )
            self.services.reporting.add_experiment(
                current_experiment.algorithms
            )
            self.services.reporting.clear_context()

    def _reset_experiment_results(self) -> None:
        """Reset experiment results tracking to empty state.
        
        This method initializes the experiment_results structure as a nested
        defaultdict that will store results organized by group name and dataset.
        Each experiment result is stored as a list of dictionaries containing
        experiment metadata, status, and timing information.
        
        Notes
        -----
        The structure created is:
        experiment_results[group_name][dataset_name] = [result1, result2, ...]
        
        Each result dictionary contains:
        - experiment: Name of the experiment
        - status: "PASSED" or "FAILED"
        - time_taken: Formatted time string
        - error: Error message (if failed)
        """
        self.experiment_results = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )

    def _create_report(self, results_dir: str) -> None:
        """Create an HTML report from the experiment results.
        
        This method generates a comprehensive HTML report containing all
        experiment results, visualizations, and analysis. It uses the
        reporting service to collect all data and the report renderer
        to generate the final HTML output.
        
        Parameters
        ----------
        results_dir : str
            Directory where the HTML report will be saved
            
        Notes
        -----
        The method uses the reporting service to collect all experiment data
        and then uses the ReportRenderer to generate the final HTML report.
        The report includes all visualizations, metrics, and analysis results
        from the completed experiments.
        """
        report_data = self.services.reporting.get_report_data()
        report_renderer.ReportRenderer().render(report_data, results_dir)

    def _setup_workflow(
        self,
        current_experiment: experiment.Experiment,
        workflow: Type[workflow_module.Workflow],
        results_dir: str
    ) -> workflow_module.Workflow:
        """Prepare a workflow instance for experiment execution.
        
        This method sets up all necessary components for running a workflow,
        including data splitting, evaluation manager configuration, and
        workflow instantiation with all required parameters.
        
        The method handles data loading, splitting, and preparation for
        the specific experiment, then creates a configured workflow instance
        ready for execution.
        
        Parameters
        ----------
        current_experiment : Experiment
            The experiment to set up, containing all configuration and data
            paths
        workflow : Type[Workflow]
            The workflow class to instantiate for this experiment
        results_dir : str
            Base directory where experiment results will be stored
            
        Returns
        -------
        Workflow
            Fully configured workflow instance ready for execution
            
        Notes
        -----
        The method performs the following setup steps:
        1. Extracts experiment metadata (group, dataset, name)
        2. Loads and splits the dataset using the appropriate data manager
        3. Creates the experiment output directory
        4. Configures the evaluation manager with experiment values
        5. Instantiates the workflow with all required parameters
        
        The workflow instance is configured with:
        - Training and test data (X_train, X_test, y_train, y_test)
        - Algorithm names to use
        - Feature names for the dataset
        - Output directory for results
        - Workflow-specific attributes
        - Evaluation manager for metrics
        """
        group_name = current_experiment.group_name
        dataset_name = current_experiment.dataset_name
        experiment_name = current_experiment.name

        data_split = self.data_managers[group_name].split(
            data_path=current_experiment.dataset_path,
            categorical_features=current_experiment.categorical_features,
            table_name=current_experiment.dataset_name[1],
            group_name=group_name,
            filename=current_experiment.dataset_name[0]
        ).get_split(current_experiment.split_index)

        X_train, X_test, y_train, y_test = data_split.get_train_test() # pylint: disable=C0103

        experiment_dir = self._get_experiment_dir(
            results_dir, group_name, dataset_name,
            current_experiment.split_index, experiment_name
        )

        self.eval_manager.set_experiment_values(
            experiment_dir, data_split.get_split_metadata(),
            data_split.group_index_train, data_split.group_index_test
        )

        workflow_instance = workflow(
            evaluation_manager=self.eval_manager,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            output_dir=experiment_dir,
            algorithm_names=current_experiment.algorithm_names,
            feature_names=data_split.features,
            workflow_attributes=current_experiment.workflow_attributes
        )
        return workflow_instance

    def _handle_success(
        self,
        start_time: float,
        group_name: str,
        experiment_name: str,
        dataset: str,
        split_index: int
    ) -> None:
        """Handle results for a successful experiment.
        
        This method processes the successful completion of an experiment,
        calculating execution time, updating result tracking, and logging
        the success with appropriate formatting.
        
        Parameters
        ----------
        start_time : float
            Time when the experiment started (from time.time())
        group_name : str
            Name of the experiment group
        dataset_name : str
            Name of the dataset (tuple format)
        experiment_name : str
            Name of the experiment
        dataset : str
            Display name of the dataset
        split_index : int
            Index of the data split used
            
        Notes
        -----
        The method:
        1. Calculates elapsed time from start_time
        2. Updates experiment_results with success status
        3. Logs success message with timing information
        4. Uses formatted time display (minutes and seconds)
        """
        elapsed_time = time.time() - start_time
        self.experiment_results[group_name][dataset].append({
            "experiment": experiment_name,
            "status": "PASSED",
            "time_taken": self._format_time(elapsed_time)
        })
        tqdm.tqdm.write(
            f"\nExperiment '{experiment_name}' on dataset "
            f"'{dataset}' (Split {split_index}) PASSED in "
            f"{self._format_time(elapsed_time)}."
        )

    def _handle_failure(
        self,
        group_name: str,
        experiment_name: str,
        start_time: float,
        error: Exception,
        dataset: str,
        split_index: int
    ) -> None:
        """Handle results and logging for a failed experiment.
        
        This method processes the failure of an experiment, calculating
        execution time, logging detailed error information, and updating
        result tracking with failure status and error details.
        
        Parameters
        ----------
        group_name : str
            Name of the experiment group
        dataset_name : str
            Name of the dataset (tuple format)
        experiment_name : str
            Name of the experiment
        start_time : float
            Time when the experiment started (from time.time())
        error : Exception
            Exception that caused the experiment to fail
        dataset : str
            Display name of the dataset
        split_index : int
            Index of the data split used
            
        Notes
        -----
        The method:
        1. Calculates elapsed time from start_time
        2. Logs detailed error information with context
        3. Updates experiment_results with failure status
        4. Logs failure message with timing information
        5. Includes error message in the result tracking
        """
        elapsed_time = time.time() - start_time
        error_message = (
            f"\n\nDataset Name: {dataset}\n"
            f"Experiment Name: {experiment_name}\n\n"
            f"Error: {error}"
        )
        self.services.logger.logger.exception(error_message)

        self.experiment_results[group_name][dataset].append({
            "experiment": experiment_name,
            "status": "FAILED",
            "time_taken": self._format_time(elapsed_time),
            "error": str(error)
        })
        tqdm.tqdm.write(
            f"\nExperiment '{experiment_name}' on dataset "
            f"'{dataset}' (Split {split_index}) FAILED in "
            f"{self._format_time(elapsed_time)}."
        )

    def _log_warning(
        self,
        message: str,
        category: Type[Warning],
        filename: str,
        lineno: int,
        dataset_name: Optional[str] = None,
        experiment_name: Optional[str] = None
    ) -> None:
        """Log warnings with specific formatting and context.
        
        This method provides custom warning logging that includes experiment
        context information, making it easier to track warnings to specific
        experiments and datasets during execution.
        
        Parameters
        ----------
        message : str
            The warning message to log
        category : Type[Warning]
            The type/category of the warning
        filename : str
            Name of the file where the warning occurred
        lineno : int
            Line number where the warning occurred
        dataset_name : str, optional
            Name of the dataset being processed, by default None
        experiment_name : str, optional
            Name of the experiment being run, by default None
            
        Notes
        -----
        The method formats warnings with:
        - Dataset and experiment context
        - File and line number information
        - Warning category and message
        - Clear separation for readability
        
        This helps with debugging by providing context about which
        experiment and dataset triggered the warning.
        """
        log_message = (
            f"\n\nDataset Name: {dataset_name} \n"
            f"Experiment Name: {experiment_name}\n\n"
            f"Warning in {filename} at line {lineno}:\n"
            f"Category: {category.__name__}\n\n"
            f"Message: {message}\n"
        )
        self.services.logger.logger.warning(log_message)

    def _print_experiment_summary(self) -> None:
        """Print experiment summary organized by group and dataset.
        
        This method displays a comprehensive summary of all experiment results
        in a formatted table, showing the status and execution time for each
        experiment, organized by experiment group and dataset.
        
        Notes
        -----
        The summary includes:
        - Group-by-group organization of results
        - Dataset-by-dataset breakdown within each group
        - Experiment name, status, and execution time
        - Clear formatting with separators and headers
        - Brisk version information at the end
        
        The output is formatted as a table with columns:
        - Experiment: Name of the experiment
        - Status: "PASSED" or "FAILED"
        - Time: Execution time in formatted format
        """
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)

        for group_name, datasets in self.experiment_results.items():
            print(f"\nGroup: {group_name}")
            print("="*70)

            for dataset_name, experiments in datasets.items():
                print(f"\nDataset: {dataset_name}")
                print(f"{'Experiment':<50} {'Status':<10} {'Time':<10}") # pylint: disable=W1405
                print("-"*70)

                for result in experiments:
                    print(
                        f"{result['experiment']:<50} {result['status']:<10} " # pylint: disable=W1405
                        f"{result['time_taken']:<10}" # pylint: disable=W1405
                    )
            print("="*70)
        print("\nModels trained using Brisk version", __version__)

    def _get_experiment_dir(
        self,
        results_dir: str,
        group_name: str,
        dataset_name: str,
        split_index: int,
        experiment_name: str
    ) -> str:
        """Create and return the directory path for experiment results.
        
        This method creates a structured directory path for storing experiment
        results, organizing them by group, dataset, split, and experiment name.
        It also creates the directory if it doesn't exist and tracks the path
        in the experiment_paths structure.
        
        Parameters
        ----------
        results_dir : str
            Base directory where all results are stored
        group_name : str
            Name of the experiment group
        dataset_name : str
            Name of the dataset (tuple format)
        split_index : int
            Index of the data split being used
        experiment_name : str
            Name of the specific experiment
            
        Returns
        -------
        str
            Full path to the experiment directory
            
        Notes
        -----
        The directory structure created is:
        results_dir/group_name/dataset_name/split_{split_index}/experiment_name/
        
        For datasets with table names, the directory name includes both
        the dataset name and table name separated by an underscore.
        
        The method also updates the experiment_paths tracking structure
        for later reference.
        """
        if dataset_name[1] is None:
            dataset_dir_name = dataset_name[0]
        else:
            dataset_dir_name = f"{dataset_name[0]}_{dataset_name[1]}"

        full_path = os.path.normpath(
            os.path.join(
                results_dir, group_name, dataset_dir_name,
                f"split_{split_index}", experiment_name
            )
        )
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        (self.experiment_paths
            [group_name][dataset_name][f"split_{split_index}"][experiment_name]
         ) = full_path

        return full_path

    def _format_time(self, seconds: float) -> str:
        """Format time taken in minutes and seconds.
        
        This method converts a time duration in seconds to a human-readable
        format showing minutes and seconds, making it easier to understand
        experiment execution times.
        
        Parameters
        ----------
        seconds : float
            Time duration in seconds
            
        Returns
        -------
        str
            Formatted time string in "Xm Ys" format
            
        Examples
        --------
        >>> trainer = TrainingManager(metric_config, config_manager)
        >>> trainer._format_time(125.5)
        '2m 5s'
        >>> trainer._format_time(60.0)
        '1m 0s'
        >>> trainer._format_time(30.0)
        '0m 30s'
        """
        mins, secs = divmod(seconds, 60)
        return f"{int(mins)}m {int(secs)}s"

    def _cleanup(self, results_dir: str, progress_bar: tqdm.tqdm) -> None:
        """Clean up resources and remove empty error log files.
        
        This method performs cleanup operations after all experiments
        have completed, including closing the progress bar and removing
        empty error log files to keep the results directory clean.
        
        Parameters
        ----------
        results_dir : str
            Directory where results are stored
        progress_bar : tqdm.tqdm
            Progress bar instance to close
            
        Notes
        -----
        The cleanup process:
        1. Closes the progress bar to free resources
        2. Checks if error_log.txt exists and is empty
        3. Removes empty error log files to keep directory clean
        
        This helps maintain a clean results directory by removing
        unnecessary empty log files.
        """
        progress_bar.close()
        error_log_path = os.path.join(results_dir, "error_log.txt")
        if (os.path.exists(error_log_path)
            and os.path.getsize(error_log_path) == 0
            ):
            os.remove(error_log_path)
