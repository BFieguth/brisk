"""Provides the TrainingManager class to manage the training of models.

Exports:
    - TrainingManager: A class to handle model training across multiple 
        datasets and methods.
"""

from collections import deque
from datetime import datetime
import itertools
import os
import traceback
from typing import List, Dict, Tuple, Callable, Optional

import pandas as pd

from brisk.data.DataSplitter import DataSplitter
from brisk.training.Workflow import Workflow
from brisk.evaluation.EvaluationManager import EvaluationManager
from brisk.reporting.ReportManager import ReportManager

class TrainingManager:
    """A class to manage the training and evaluation of machine learning models.

    The TrainingManager coordinates the training of models using various methods, 
    evaluates them on different datasets, and generates reports. It integrates with 
    EvaluationManager for model evaluation and ReportManager for generating HTML reports.

    Attributes:
        method_config (dict): Configuration of methods with default parameters.
        scoring_config (dict): Configuration of scoring metrics.
        workflow (Workflow): An instance of the Workflow class to define training steps.
        splitter (DataSplitter): Instance of the DataSplitter class for train-test splits.
        methods (list): List of methods to apply to each dataset.
        data_paths (list): List of tuples containing dataset paths and table names.
        results_dir (str, optional): Directory to store results. Defaults to None.
        EvaluationManager (EvaluationManager): Instance of the EvaluationManager 
            class for handling evaluations.
    """
    def __init__(
        self, 
        method_config: Dict[str, Dict], 
        scoring_config: Dict[str, Dict], 
        workflow: Workflow,
        splitter: DataSplitter, 
        methods: List[str], 
        data_paths: List[Tuple[str, str]],
        results_dir: Optional[str] = None
    ):
        """Initializes the TrainingManager.

        Args:
            method_config (Dict[str, Dict]): Configuration of methods with default parameters.
            scoring_config (Dict[str, Dict]): Configuration of scoring metrics.
            splitter (DataSplitter): An instance of the DataSplitter class for train-test splits.
            workflow (Workflow): An instance of the Workflow class to define training steps.
            methods (List[str]): List of methods to train on each dataset.
            data_paths (List[Tuple[str, str]]): List of tuples containing dataset paths and table names.
            results_dir (Optional[str]): Directory to store results. If None, a timestamp will be used.
        """
        self.method_config = method_config
        self.scoring_config = scoring_config
        self.workflow = workflow
        self.splitter = splitter
        self.methods = methods
        self.data_paths = data_paths

        if results_dir:
            if os.path.exists(results_dir):
                raise FileExistsError(
                    f"Results directory '{results_dir}' already exists."
                    )
            self.results_dir = results_dir
        else:
            self.results_dir = None

        self.EvaluationManager = EvaluationManager(
            method_config=self.method_config, scoring_config=self.scoring_config
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
        """Splits each dataset using the provided splitter.
         
        Returns a dictionary mapping each dataset path to its respective 
        train-test splits.

        Returns:
            Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]: 
            A dictionary where the key is the dataset path, and the value is the train-test split.
        """
        data_splits = {}
        for data_path, table_name in self.data_paths:
            X_train, X_test, y_train, y_test = self.splitter.split(
                data_path, table_name
                )
            data_splits[data_path] = (X_train, X_test, y_train, y_test)
        
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
        return f"{timestamp}_Results"

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
        experiment_dir = f"{method_name}_{dataset_name}"
        full_path = os.path.join(results_dir, experiment_dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        return full_path

    def run_experiments(
        self, 
        create_report: bool = True
    ) -> None:
        """Runs the user-defined workflow for each experiment and optionally generates reports.

        Args:
            create_report (bool): Whether to generate an HTML report after all 
                experiments. Defaults to True.

        Returns:
            None
        """
        error_log = []
        self.experiment_paths = {}
        if not self.results_dir:
            results_dir = self._get_results_dir()
        else:
            results_dir = self.results_dir

        os.makedirs(results_dir, exist_ok=True)

        while self.experiments:
            data_path, method_names = self.experiments.popleft()
            try:
                X_train, X_test, y_train, y_test = self.data_splits[data_path[0]]

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
                    output_dir=experiment_dir
                    )
                workflow_instance = self.workflow(
                    evaluator=config_EvaluationManager,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    output_dir=experiment_dir,
                    method_names=method_names,
                    model_kwargs=model_kwargs
                )

                # Call the workflow method
                workflow_instance.workflow()

            except Exception as e:
                error_message = f"Error for {method_names} on {data_path}: {str(e)}"
                print(error_message)
                traceback.print_exc()
                error_log.append(error_message)
        
        if error_log:
            error_log_path = os.path.join(results_dir, "error_log.txt")
            with open(error_log_path, "w") as file:
                file.write("\n".join(error_log))
                
        if create_report:
            report_manager = ReportManager(results_dir, self.experiment_paths)
            report_manager.create_report()
