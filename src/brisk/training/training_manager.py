"""Provides the TrainingManager class to manage the training of models.

This module defines the TrainingManager class, which coordinates model training
across multiple datasets and algorithms. Ensures all models are attempted even
if some fail.
"""

import collections
from datetime import datetime
import logging
import os
import pathlib
import time
from typing import Dict, Tuple, Optional, Type
import warnings
import re

import joblib
import tqdm

from brisk.evaluation import evaluation_manager, metric_manager

from brisk.reporting import report_renderer

from brisk.training import logging_util
from brisk.configuration import algorithm_wrapper, configuration
from brisk.version import __version__
from brisk.training import workflow as workflow_module
from brisk.configuration import experiment
from brisk.services import get_services

class TrainingManager:
    """Manage the training and evaluation of machine learning models.

    Coordinates model training using various algorithms, evaluates them on 
    different datasets, and generates reports. Integrates with EvaluationManager
    for model evaluation and ReportManager for generating HTML reports.

    Parameters
    ----------
    metric_config : MetricManager
        Configuration for evaluation metrics
    config_manager : ConfigurationManager
        Instance containing data needed to run experiments

    Attributes
    ----------
    metric_config : MetricManager
        Configuration for evaluation metrics
    data_managers : dict
        Maps group names to their data managers
    experiments : collections.deque
        Queue of experiments to run
    logfile : str
        Path to the configuration log file
    output_structure : dict
        Structure of output data organization
    description_map : dict
        Mapping of names to descriptions
    experiment_paths : defaultdict
        Nested structure tracking experiment output paths
    experiment_results : defaultdict
        Stores results of all experiments
    """
    def __init__(
        self,
        metric_config: metric_manager.MetricManager,
        config_manager: configuration.ConfigurationManager,
    ):
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
        workflow: workflow_module.Workflow,
        # results_name: Optional[str] = None,
        create_report: bool = True
    ) -> None:
        """Runs the Workflow for each experiment and generates report.

        Parameters
        ----------
        workflow : Workflow
            A subclass of the Workflow class that defines the training steps.

        results_name : str
            The name of the results directory.

        create_report : bool
            Whether to generate an HTML report after all experiments. 
            Defaults to True.
        """
        self._reset_experiment_results()
        progress_bar = tqdm.tqdm(
            total=len(self.experiments),
            desc="Running Experiments",
            unit="experiment"
        )

        # results_dir = self._create_results_dir(results_name)
        self._save_config_log(self.results_dir, workflow, self.logfile)
        # self._save_data_distributions(self.results_dir, self.output_structure)

        while self.experiments:
            current_experiment = self.experiments.popleft()
            self._run_single_experiment(
                current_experiment,
                workflow,
                self.results_dir
            )
            progress_bar.update(1)

        self._print_experiment_summary()
        self._cleanup(self.results_dir, progress_bar)
        if create_report:
            self._create_report(self.results_dir)

    def _run_single_experiment(
        self,
        current_experiment: experiment.Experiment,
        workflow: workflow_module.Workflow,
        results_dir: str
    ) -> None:
        """Runs a single Experiment and handles its outcome.

        Sets up the experiment environment, runs the workflow, and handles 
        success or failure cases.

        Parameters
        ----------
        current_experiment : Experiment
            The experiment to run.

        workflow : Workflow
            The workflow to use.

        results_dir : str
            Directory to store results.
        """
        success = False
        start_time = time.time()

        group_name = current_experiment.group_name
        dataset_name = current_experiment.dataset_name
        experiment_name = current_experiment.name

        tqdm.tqdm.write(f"\n{'=' * 80}") # pylint: disable=W1405
        tqdm.tqdm.write(
            f"\nStarting experiment '{experiment_name}' on dataset "
            f"'{dataset_name}'."
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
                current_experiment, workflow, results_dir, group_name,
                dataset_name, experiment_name
            )
            workflow_instance.workflow()
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
                dataset_name,
                experiment_name,
                start_time,
                e
            )

        if success:
            self._handle_success(
                start_time,
                group_name,
                dataset_name,
                experiment_name
            )

    def _reset_experiment_results(self) -> None:
        """Set self.experiment_results to a defaultdict of lists."""
        self.experiment_results = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )

    def _create_report(self, results_dir: str) -> None:
        """Create an HTML report from the experiment results.

        Parameters
        ----------
        results_dir : str
            Directory where results are stored.
        """
        report_data = self.services.reporting.get_report_data()
        report_renderer.ReportRenderer().render(report_data, results_dir)

    def _save_config_log(
        self,
        results_dir: str,
        workflow: workflow_module.Workflow,
        logfile: str
    ) -> None:
        """Saves the workflow configuration and class name to a config log file.

        Parameters
        ----------
        results_dir : str
            Directory where results are stored.

        workflow : Workflow
            The workflow to save.

        logfile : str
            The logfile to save.
        """
        config_log_path = os.path.join(results_dir, "config_log.md")
        config_content = logfile.split("\n")
        workflow_md = [
            "# Experiment Configuration Log",
            "",
            "## Workflow Configuration",
            "",
            f"### Workflow Class: `{workflow.__name__}`",
            ""
        ]

        full_content = "\n".join(workflow_md + config_content)

        with open(config_log_path, "w", encoding="utf-8") as f:
            f.write(full_content)

    def _setup_workflow(
        self,
        current_experiment: experiment.Experiment,
        workflow: Type[workflow_module.Workflow],
        results_dir: str,
        group_name: str,
        dataset_name: str,
        experiment_name: str
    ) -> workflow_module.Workflow:
        """Prepares a workflow instance for experiment execution.

        Sets up data, algorithms, and evaluation manager for the workflow.

        Parameters
        ----------
        current_experiment : Experiment
            The experiment to set up.

        workflow : Workflow
            The workflow to instantiate.

        results_dir : str
            Directory for results.

        group_name : str
            Name of the experiment group.

        dataset_name : str
            Name of the dataset.

        experiment_name : str
            Name of the experiment.

        Returns
        -------
        Workflow
            Configured workflow instance.
        """
        data_split = self.data_managers[group_name].split(
            data_path=current_experiment.dataset_path,
            categorical_features=current_experiment.categorical_features,
            table_name=current_experiment.table_name,
            group_name=group_name,
            filename=dataset_name
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
        dataset_name: str,
        experiment_name: str
    ) -> None:
        """Handle results for a successful experiment.

        Parameters
        ----------
        start_time : float
            Time when experiment started
        group_name : str
            Name of experiment group
        dataset_name : str
            Name of dataset
        experiment_name : str
            Name of experiment
        """
        elapsed_time = time.time() - start_time
        self.experiment_results[group_name][dataset_name].append({
            "experiment": experiment_name,
            "status": "PASSED",
            "time_taken": self._format_time(elapsed_time)
        })
        tqdm.tqdm.write(
            f"\nExperiment '{experiment_name}' on dataset "
            f"'{dataset_name}' PASSED in {self._format_time(elapsed_time)}."
        )
        tqdm.tqdm.write(f"\n{'-' * 80}") # pylint: disable=W1405

    def _handle_failure(
        self,
        group_name: str,
        dataset_name: str,
        experiment_name: str,
        start_time: float,
        error: Exception
    ) -> None:
        """Handle results and logging for a failed experiment.

        Parameters
        ----------
        group_name : str
            Name of experiment group
        dataset_name : str
            Name of dataset
        experiment_name : str
            Name of experiment
        start_time : float
            Time when experiment started
        error : Exception
            Exception that caused the failure
        """
        elapsed_time = time.time() - start_time
        error_message = (
            f"\n\nDataset Name: {dataset_name}\n"
            f"Experiment Name: {experiment_name}\n\n"
            f"Error: {error}"
        )
        self.services.logger.logger.exception(error_message)

        self.experiment_results[group_name][dataset_name].append({
            "experiment": experiment_name,
            "status": "FAILED",
            "time_taken": self._format_time(elapsed_time),
            "error": str(error)
        })
        tqdm.tqdm.write(
            f"\nExperiment '{experiment_name}' on dataset "
            f"'{dataset_name}' FAILED in {self._format_time(elapsed_time)}."
        )
        tqdm.tqdm.write(f"\n{'-' * 80}") # pylint: disable=W1405

    def _log_warning(
        self,
        message: str,
        category: Type[Warning],
        filename: str,
        lineno: int,
        dataset_name: Optional[str] = None,
        experiment_name: Optional[str] = None
    ) -> None:
        """Log warnings with specific formatting.

        Parameters
        ----------
        message : str
            Warning message
        category : Type[Warning]
            Warning category
        filename : str
            File where warning occurred
        lineno : int
            Line number where warning occurred
        dataset_name : str, optional
            Name of dataset, by default None
        experiment_name : str, optional
            Name of experiment, by default None
        """
        log_message = (
            f"\n\nDataset Name: {dataset_name} \n"
            f"Experiment Name: {experiment_name}\n\n"
            f"Warning in {filename} at line {lineno}:\n"
            f"Category: {category.__name__}\n\n"
            f"Message: {message}\n"
        )
        logger = logging.getLogger("TrainingManager")
        logger.warning(log_message)

    def _print_experiment_summary(self) -> None:
        """Print experiment summary organized by group and dataset.

        Displays a formatted table showing the status and execution time
        for each experiment, grouped by dataset and experiment group.
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
        """Creates and returns the directory path for experiment results.

        Parameters
        ----------
        results_dir : str
            Base results directory.

        group_name : str
            Name of the experiment group.

        dataset_name : str
            Name of the dataset.

        experiment_name : str
            Name of the experiment.

        Returns
        -------
        str
            Path to the experiment directory.
        """
        full_path = os.path.normpath(
            os.path.join(
                results_dir, group_name, dataset_name, f"split_{split_index}", 
                experiment_name
            )
        )
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        (self.experiment_paths
            [group_name][dataset_name][f"split_{split_index}"][experiment_name]
         ) = full_path

        return full_path

    def _format_time(self, seconds: float) -> str:
        """Formats time taken in minutes and seconds.

        Parameters
        ----------
        seconds : float
            Time taken in seconds.

        Returns
        -------
        str
            Formatted time string.
        """
        mins, secs = divmod(seconds, 60)
        return f"{int(mins)}m {int(secs)}s"

    def _cleanup(self, results_dir: str, progress_bar: tqdm.tqdm) -> None:
        """Shuts down logging and deletes error_log.txt if it is empty.

        Parameters
        ----------
        results_dir : str
            Directory where results are stored.

        progress_bar : tqdm.tqdm
            Progress bar to close.
        """
        progress_bar.close()
        logging.shutdown()
        error_log_path = os.path.join(results_dir, "error_log.txt")
        if (os.path.exists(error_log_path)
            and os.path.getsize(error_log_path) == 0
            ):
            os.remove(error_log_path)
