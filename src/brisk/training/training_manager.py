"""Provides the TrainingManager class to manage the training of models.

Exports:
    - TrainingManager: A class to handle model training across multiple 
        datasets and algorithms.
"""

import collections
from datetime import datetime
import logging
import os
import time
from typing import Dict, Tuple
import warnings

import joblib
from tqdm import tqdm

from brisk.evaluation import evaluation_manager
from brisk.reporting import report_manager as report
from brisk.utility import logging_util
from brisk.configuration import configuration
from brisk.utility import utility
from brisk.version import __version__

class TrainingManager:
    """A class to manage the training and evaluation of machine learning models.

    The TrainingManager coordinates the training of models using various 
    algorithms, evaluates them on different datasets, and generates reports. 
    It integrates with EvaluationManager for model evaluation and ReportManager
    for generating HTML reports.

    Attributes:
        metric_config (lis): Configuration of scoring metrics.

        DataManager (DataManager): Instance of the DataManager class for 
        train-test splits.

        algorithms (list): List of algorithms to apply to each dataset.

        data_paths (list): List of tuples containing dataset paths and table 
        names.

        results_dir (str, optional): Directory to store results. Defaults to 
        None.

        EvaluationManager (EvaluationManager): Instance of the EvaluationManager
        class for handling evaluations.
    """
    def __init__(
        self,
        metric_config: Dict[str, Dict],
        config_manager: configuration.ConfigurationManager,
        verbose=False
    ):
        """Initializes the TrainingManager.

        Args:
            metric_config (Dict[str, Dict]): Configuration of scoring metrics.

            config_manager (ConfigurationManager): Instance of 
            ConfigurationManager with data needed to run experiments
        """
        self.metric_config = metric_config
        self.verbose = verbose
        self.data_managers = config_manager.data_managers
        self.experiments = config_manager.experiment_queue
        self.logfile = config_manager.logfile
        self.output_structure = config_manager.output_structure
        self.description_map = config_manager.description_map
        self.experiment_paths = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: {}
            )
        )

    def _get_results_dir(self) -> str:
        """Generates a results directory name based on the current timestamp.

        Returns:
            str: The directory name for storing results.
        """
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        results_dir = os.path.join("results", timestamp)
        return results_dir

    def _get_experiment_dir(
        self,
        results_dir: str,
        group_name: str,
        dataset_name: str,
        experiment_name: str
    ) -> str:
        """Creates a meaningful directory name for storing experiment results.

        Args:
            method_name (str): The name of the method being used.

            data_path (Tuple[str, str]): The dataset path and table name 
            (if applicable).
            
            results_dir (str): The root directory for storing results.

        Returns:
            str: The full path to the directory for storing experiment results.
        """
        full_path = os.path.normpath(
            os.path.join(results_dir, group_name, dataset_name, experiment_name)
        )
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        return full_path

    def run_experiments(
        self,
        workflow,
        workflow_config = None,
        results_name = None,
        create_report: bool = True
    ) -> None:
        """Runs the user-defined workflow for each experiment and optionally 
        generates reports.

        Args:
            workflow (Workflow): An instance of the Workflow class to define 
            training steps.
            
            workflow_config: Variables to pass the workflow class.
            
            create_report (bool): Whether to generate an HTML report after all 
            experiments. Defaults to True.

        Returns:
            None
        """
        def format_time(seconds):
            mins, secs = divmod(seconds, 60)
            return f"{int(mins)}m {int(secs)}s"


        def log_warning(
            message,
            category,
            filename,
            lineno,
            dataset_name=None,
            experiment_name=None
        ):
            """
            Custom warning handler that logs warnings with specific formatting.
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


        logging.captureWarnings(True)
        experiment_results = collections.defaultdict(
            lambda: collections.defaultdict(
                list
            )
        )

        if not results_name:
            results_dir = self._get_results_dir()
        else:
            results_dir = os.path.join("results", results_name)
            if os.path.exists(results_dir):
                raise FileExistsError(
                    f"Results directory '{results_dir}' already exists."
                )
        os.makedirs(results_dir, exist_ok=False)

        self._save_config_log(
            results_dir, workflow, workflow_config, self.logfile
            )
        self._save_data_distributions(results_dir, self.output_structure)
        self.logger = self._setup_logger(results_dir)
        pbar = tqdm(
            total=len(self.experiments),
            desc="Running Experiments",
            unit="experiment"
        )

        while self.experiments:
            current_experiment = self.experiments.popleft()
            group_name = current_experiment.group_name
            dataset_name = current_experiment.dataset.stem
            experiment_name = current_experiment.name

            warnings.showwarning = (
                lambda message, category, filename, lineno, file=None, line=None: log_warning( # pylint: disable=line-too-long
                    message,
                    category,
                    filename,
                    lineno,
                    dataset_name,
                    experiment_name
                )
            )

            tqdm.write(f"\n{'=' * 80}") # pylint: disable=W1405
            tqdm.write(
                f"\nStarting experiment '{experiment_name}' on dataset "
                f"'{dataset_name}'."
            )

            start_time = time.time()

            try:
                data_split = self.data_managers[group_name].split(
                    data_path=current_experiment.dataset,
                    group_name=group_name,
                    filename=dataset_name
                )
                X_train, X_test, y_train, y_test = data_split.get_train_test() # pylint: disable=C0103
                algo_kwargs = {
                    key: algo.instantiate()
                    for key, algo in current_experiment.algorithms.items()
                }
                algo_names = [
                    algo.name
                    for algo in current_experiment.algorithms.values()
                ]

                experiment_dir = self._get_experiment_dir(
                    results_dir, group_name, dataset_name, experiment_name
                )

                # Save each experiment_dir for reporting
                (self.experiment_paths
                 [group_name]
                 [dataset_name]
                 [experiment_name]) = experiment_dir

                eval_manager = evaluation_manager.EvaluationManager(
                    list(current_experiment.algorithms.values()),
                    self.metric_config,
                    experiment_dir,
                    data_split.get_split_metadata(),
                    self.logger
                )

                workflow_instance = workflow(
                    evaluator=eval_manager,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    output_dir=experiment_dir,
                    algorithm_names=algo_names,
                    feature_names=data_split.features,
                    model_kwargs=algo_kwargs,
                    workflow_config=workflow_config
                )

                start_time = time.time()
                workflow_instance.workflow()
                end_time = time.time()
                elapsed_time = end_time - start_time

                experiment_results[group_name][dataset_name].append({
                    "experiment": experiment_name,
                    "status": "PASSED",
                    "time_taken": format_time(elapsed_time)
                })
                tqdm.write(
                    f"\nExperiment '{experiment_name}' on dataset "
                    f"'{dataset_name}' PASSED in {format_time(elapsed_time)}."
                )
                tqdm.write(f"\n{'-' * 80}") # pylint: disable=W1405
                pbar.update(1)

            # TODO (Issue #68): refactor to avoid bare except here.
            # Should also reduce the size of the try block.
            except Exception as e:
                end_time = time.time()
                elapsed_time = end_time - start_time
                error_message = (
                    f"\n\nDataset Name: {dataset_name}\n"
                    f"Experiment Name: {experiment_name}\n\n"
                    f"Error: {e}"
                )
                self.logger.exception(error_message)

                experiment_results[group_name][dataset_name].append({
                    "experiment": experiment_name,
                    "status": "FAILED",
                    "time_taken": format_time(elapsed_time),
                    "error": str(e)
                })
                tqdm.write(
                    f"\nExperiment '{experiment_name}' on dataset "
                    f"'{dataset_name}' FAILED in {format_time(elapsed_time)}."
                )
                tqdm.write(f"\n{'-' * 80}") # pylint: disable=W1405
                pbar.update(1)

        pbar.close()
        self._print_experiment_summary(experiment_results)

        # Delete error_log.txt if it is empty
        logging.shutdown()
        error_log_path = os.path.join(results_dir, "error_log.txt")
        if (os.path.exists(error_log_path)
            and os.path.getsize(error_log_path) == 0
            ):
            os.remove(error_log_path)

        if create_report:
            report_manager = report.ReportManager(
                results_dir, self.experiment_paths, self.output_structure,
                self.description_map
                )
            report_manager.create_report()

    def _print_experiment_summary(self, experiment_results):
        """Print the experiment summary organized by group and dataset.
        """
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)

        for group_name, datasets in experiment_results.items():
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

    def _setup_logger(self, results_dir):
        """Set up logging for the TrainingManager.

        Logs to both file and console, using different levels for each        
        """
        logger = logging.getLogger("TrainingManager")
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(
            os.path.join(results_dir, "error_log.txt")
        )
        file_handler.setLevel(logging.WARNING)

        console_handler = logging_util.TqdmLoggingHandler()
        if self.verbose:
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(logging.ERROR)

        formatter = logging.Formatter(
            "\n%(asctime)s - %(levelname)s - %(message)s"
        )
        file_formatter = logging_util.FileFormatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _save_config_log(self, results_dir, workflow, workflow_config, logfile):
        """Saves the workflow configuration and class name to a config log file.
        """
        config_log_path = os.path.join(results_dir, "config_log.md")

        # Split the logfile back into a list
        config_content = logfile.split("\n")

        # Create the header content
        workflow_md = [
            "# Experiment Configuration Log",
            "",
            "## Workflow Configuration",
            "",
            f"### Workflow Class: `{workflow.__name__}`",
            ""
        ]

        if workflow_config:
            workflow_md.extend([
                "### Configuration:",
                "```python",
                utility.format_dict(workflow_config),
                "```",
                ""
            ])

        full_content = "\n".join(workflow_md + config_content)

        with open(config_log_path, "w", encoding="utf-8") as f:
            f.write(full_content)

    def _save_data_distributions(
        self,
        result_dir: str,
        output_structure: Dict[str, Dict[str, Tuple[str, str]]]
    ) -> None:
        """Save data distribution information for each dataset.
        
        Args:
            result_dir: Base directory for results
            output_structure: Mapping of groups to their datasets and split info
        """
        for group_name, datasets in output_structure.items():
            group_dir = os.path.join(result_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)
            group_data_manager = self.data_managers[group_name]

            for dataset_name, (data_path, group_name) in datasets.items():
                split_info = group_data_manager.split(
                    data_path=data_path,
                    group_name=group_name,
                    filename=dataset_name
                )

                dataset_dir = os.path.join(group_dir, dataset_name)
                os.makedirs(dataset_dir, exist_ok=True)

                split_info.save_distribution(
                    os.path.join(dataset_dir, "split_distribution")
                    )

                if hasattr(split_info, "scaler") and split_info.scaler:
                    split_name = split_info.scaler.__class__.__name__
                    scaler_path = os.path.join(
                        dataset_dir,
                        f"{dataset_name}_{split_name}.joblib"
                    )
                    joblib.dump(split_info.scaler, scaler_path)
