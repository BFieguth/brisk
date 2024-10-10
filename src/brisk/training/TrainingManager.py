"""Provides the TrainingManager class to manage the training of models.

Exports:
    - TrainingManager: A class to handle model training across multiple 
        datasets and methods.
"""

from collections import deque
from datetime import datetime
import itertools
import logging
import os
import time
import traceback
from typing import List, Dict, Tuple, Callable, Optional
import warnings
 
import joblib
import pandas as pd
from tqdm import tqdm

from brisk.data.DataManager import DataManager
from brisk.data.DataSplitInfo import DataSplitInfo
from brisk.training.Workflow import Workflow
from brisk.evaluation.EvaluationManager import EvaluationManager
from brisk.reporting.ReportManager import ReportManager
from brisk.utility.logging import TqdmLoggingHandler, FileFormatter

class TrainingManager:
    """A class to manage the training and evaluation of machine learning models.

    The TrainingManager coordinates the training of models using various methods, 
    evaluates them on different datasets, and generates reports. It integrates with 
    EvaluationManager for model evaluation and ReportManager for generating HTML reports.

    Attributes:
        method_config (dict): Configuration of methods with default parameters.
        metric_config (dict): Configuration of scoring metrics.
        DataManager (DataManager): Instance of the DataManager class for train-test splits.
        methods (list): List of methods to apply to each dataset.
        data_paths (list): List of tuples containing dataset paths and table names.
        results_dir (str, optional): Directory to store results. Defaults to None.
        EvaluationManager (EvaluationManager): Instance of the EvaluationManager 
            class for handling evaluations.
    """
    def __init__(
        self, 
        method_config: Dict[str, Dict], 
        metric_config: Dict[str, Dict], 
        data_manager: DataManager, 
        methods: List[str], 
        data_paths: List[Tuple[str, str]],
        verbose=False
    ):
        """Initializes the TrainingManager.

        Args:
            method_config (Dict[str, Dict]): Configuration of methods with default parameters.
            metric_config (Dict[str, Dict]): Configuration of scoring metrics.
            data_manager (DataManager): An instance of the DataManager class for train-test splits.
            workflow (Workflow): An instance of the Workflow class to define training steps.
            methods (List[str]): List of methods to train on each dataset.
            data_paths (List[Tuple[str, str]]): List of tuples containing dataset paths and table names.
        """
        self.method_config = method_config
        self.metric_config = metric_config
        self.DataManager = data_manager
        self.methods = methods
        self.data_paths = data_paths
        self.verbose = verbose
        self.EvaluationManager = EvaluationManager(
            method_config=self.method_config, metric_config=self.metric_config
        )
        self._validate_methods()
        self.data_splits = self._get_data_splits()
        self.experiments = self._create_experiments()

    def _validate_methods(self) -> None:
        """Validates that all specified methods are included in the method configuration.

        Raises:
            ValueError: If any methods are missing from the method configuration.
        """
        included_methods = self.method_config.keys()

        if any(isinstance(m, list) for m in self.methods):
            flat_methods = set(m for sublist in self.methods for m in sublist)
        else:
            flat_methods = set(self.methods)

        if flat_methods.issubset(included_methods):
            return True
        else:
            invalid_methods = list(flat_methods - set(included_methods))
            raise ValueError(
                "The following methods are not included in the configuration: "
                f"{invalid_methods}"
                )

    def _get_data_splits(
        self
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """Splits each dataset using the provided DataManager.
         
        Returns a dictionary mapping each dataset path to its respective 
        train-test splits.

        Returns:
            Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]: 
            A dictionary where the key is the dataset path, and the value is the train-test split.
        """
        data_splits = {}
        for data_path, table_name in self.data_paths:
            X_train, X_test, y_train, y_test, scaler, feature_names = self.DataManager.split(
                data_path, table_name
                )

            filename = table_name if table_name else os.path.splitext(
                os.path.basename(data_path)
                )[0]
            
            data_splits[data_path] = DataSplitInfo(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                filename=filename,
                scaler=scaler,
                features=feature_names,
                categorical_features=self.DataManager.categorical_features
            )
        
        return data_splits

    def _create_experiments(self) -> List[Tuple[str, str]]:
        """Creates experiments as a Cartesian product of methods and datasets.

        Returns:
            List[Tuple[str, str]]: A list of tuples where each tuple 
                represents (data_path, method).
        """
        if all(isinstance(method, str) for method in self.methods):
            method_combinations = [(method,) for method in self.methods]
        else:
            method_combinations = zip(*self.methods)

        experiments = deque(itertools.product(self.data_paths, method_combinations))
        return experiments

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
        method_name: str, 
        data_path: Tuple[str, str],
        results_dir: str
    ) -> str:
        """Creates a meaningful directory name for storing experiment results.

        Args:
            method_name (str): The name of the method being used.
            data_path (Tuple[str, str]): The dataset path and table name (if applicable).
            results_dir (str): The root directory for storing results.

        Returns:
            str: The full path to the directory for storing experiment results.
        """
        dataset_name = os.path.basename(data_path[0]).split(".")[0]

        # experiment_dir = f"{method_name}_{dataset_name}"
        # full_path = os.path.join(results_dir, experiment_dir)

        experiment_path = f"{dataset_name}/{method_name}"
        full_path = os.path.join(results_dir, experiment_path)

        if not os.path.exists(full_path):
            os.makedirs(full_path)
        return full_path   

    def _save_scalers(self, results_dir):
        scaler_dir = os.path.join(results_dir, "scalers")
        os.makedirs(scaler_dir, exist_ok=True)
        
        for data_split in self.data_splits.values():
            if data_split.scaler:
                scaler_path = os.path.join(scaler_dir, f"{data_split.filename}_scaler.pkl")
                joblib.dump(data_split.scaler, scaler_path)

    def run_experiments(
        self, 
        workflow,
        workflow_config = None,
        results_name = None,
        create_report: bool = True
    ) -> None:
        """Runs the user-defined workflow for each experiment and optionally generates reports.

        Args:
            workflow (Workflow): An instance of the Workflow class to define training steps.
            workflow_config: Variables to pass the workflow class.
            create_report (bool): Whether to generate an HTML report after all 
                experiments. Defaults to True.

        Returns:
            None
        """
        def format_time(seconds):
            mins, secs = divmod(seconds, 60)
            return f"{int(mins)}m {int(secs)}s"


        def log_warning(message, category, filename, lineno, file=None, line=None, dataset_name=None, experiment_name=None):
            """Custom warning handler that logs warnings with specific formatting."""
            log_message = (
                f"\n\nDataset Name: {dataset_name} \nExperiment Name: {experiment_name}\n\n"
                f"Warning in {filename} at line {lineno}:\n"
                f"Category: {category.__name__}\n\n"
                f"Message: {message}\n"
            )
            logger = logging.getLogger("TrainingManager")
            logger.warning(log_message)


        def save_config_log(results_dir, workflow, workflow_config, DataManager, data_splits):
            """Saves the workflow configuration and class name to a config log file."""
            config_log_path = os.path.join(results_dir, "config_log.txt")
            
            with open(config_log_path, 'w') as f:
                f.write(f"Workflow Class: {workflow.__name__}\n\n")
                
                f.write("Workflow Configuration:\n")
                if workflow_config:
                    for key, value in workflow_config.items():
                        f.write(f"{key}: {value}\n")
                else:
                    f.write("No workflow configuration provided.\n")

                f.write("\nDataManager Configuration:\n")
                for attr, value in vars(DataManager).items():
                    f.write(f"{attr}: {value}\n")

                f.write("\nFeature Order:\n")
                for data_split in data_splits.values():
                        f.write(f"Filename: {data_split.filename}\n")
                        f.write(f"Feature Order: {', '.join(data_split.features)}\n\n")

        logging.captureWarnings(True)

        self.experiment_paths = {}
        experiment_results = {}
        total_experiments = len(self.experiments)

        if not results_name:
            results_dir = self._get_results_dir()
        else:
            results_dir = os.path.join("results", results_name)
            if os.path.exists(results_dir):
                raise FileExistsError(
                    f"Results directory '{results_dir}' already exists."
                    )

        os.makedirs(results_dir, exist_ok=True)

        save_config_log(
            results_dir, workflow, workflow_config, self.DataManager, 
            self.data_splits
            )

        self._save_scalers(results_dir)

        self.logger = self._setup_logger(results_dir)
         
        pbar = tqdm(total=total_experiments, desc="Running Experiments", unit="experiment")

        while self.experiments:
            data_path, method_names = self.experiments.popleft()
            dataset_name = os.path.basename(data_path[0])
            experiment_name = f"{'_'.join(method_names)}"

            warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: log_warning(
                message, category, filename, lineno, file, line, dataset_name, experiment_name
            )

            if dataset_name not in experiment_results:
                experiment_results[dataset_name] = []

            tqdm.write(f"\n{"-" * 80}")
            tqdm.write(f"\nStarting experiment '{experiment_name}' on dataset '{dataset_name}'.")

            try:
                data_split = self.data_splits[data_path[0]]
                X_train, X_test, y_train, y_test = data_split.get_train_test()

                models = [
                    self.method_config[method_name].instantiate() 
                    for method_name in method_names
                    ]
                if len(models) == 1:
                    model_kwargs = {"model": models[0]}
                else:
                    model_kwargs = {
                        f"model{i+1}": model for i, model in enumerate(models)
                        }

                experiment_dir = self._get_experiment_dir(
                    "_".join(method_names), data_path, results_dir
                    )
                # Save each experiment_dir for reporting, grouped by dataset
                if data_path[0] in self.experiment_paths:
                    self.experiment_paths[data_path[0]].append(experiment_dir)
                else:
                    self.experiment_paths[data_path[0]] = [experiment_dir]

                config_EvaluationManager = self.EvaluationManager.with_config(
                    output_dir=experiment_dir, logger=self.logger
                    )
                workflow_instance = workflow(
                    evaluator=config_EvaluationManager,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    output_dir=experiment_dir,
                    method_names=method_names,
                    model_kwargs=model_kwargs,
                    workflow_config=workflow_config
                )

                start_time = time.time()
                workflow_instance.workflow()
                end_time = time.time()
                elapsed_time = end_time - start_time

                experiment_results[dataset_name].append({
                    "experiment": experiment_name,
                    "status": "PASSED",
                    "time_taken": format_time(elapsed_time)
                })
                tqdm.write(f"\nExperiment '{experiment_name}' on dataset '{dataset_name}' PASSED in {format_time(elapsed_time)}.")
                tqdm.write(f"\n{"-" * 80}")
                pbar.update(1)

            except Exception as e:
                end_time = time.time()
                elapsed_time = end_time - start_time
                error_message = (
                    f"\n\nDataset Name: {dataset_name}\n"
                    f"Experiment Name: {experiment_name}\n\n"
                    f"Error: {e}"
                )
                self.logger.exception(error_message)

                experiment_results[dataset_name].append({
                    "experiment": experiment_name,
                    "status": "FAILED",
                    "time_taken": format_time(elapsed_time),
                    "error": str(e)
                })
                tqdm.write(f"\nExperiment '{experiment_name}' on dataset '{dataset_name}' FAILED in {format_time(elapsed_time)}.")
                tqdm.write(f"\n{"-" * 80}")
                pbar.update(1)
            
        pbar.close()
        self._print_experiment_summary(experiment_results)
        print(self.experiment_paths)

        # Delete error_log.txt if it is empty
        logging.shutdown()
        error_log_path = os.path.join(results_dir, "error_log.txt")
        if os.path.exists(error_log_path) and os.path.getsize(error_log_path) == 0:
            os.remove(error_log_path)

        if create_report:
            report_manager = ReportManager(results_dir, self.experiment_paths)
            report_manager.create_report()

    def _print_experiment_summary(self, experiment_results):
        """
        Print the experiment summary organized by dataset.
        """
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        for dataset_name, experiments in experiment_results.items():
            print(f"\nDataset: {dataset_name}")
            print(f"{'Experiment':<50} {'Status':<10} {'Time (MM:SS)':<10}")
            print("-"*70)
            for result in experiments:
                print(f"{result['experiment']:<50} {result['status']:<10} {result['time_taken']:<10}")
        print("="*70)

    def _setup_logger(self, results_dir):
        """Set up logging for the TrainingManager.

        Logs to both file and console, using different levels for each        
        """ 
        logger = logging.getLogger("TrainingManager")
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(os.path.join(results_dir, "error_log.txt"))
        file_handler.setLevel(logging.WARNING)

        console_handler = TqdmLoggingHandler()
        if self.verbose:
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(logging.ERROR)

        formatter = logging.Formatter("\n%(asctime)s - %(levelname)s - %(message)s")
        file_formatter = FileFormatter("%(asctime)s - %(levelname)s - %(message)s")
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
