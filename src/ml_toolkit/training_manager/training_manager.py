from collections import deque
from datetime import datetime
import itertools
import os
import traceback
from typing import List, Dict, Tuple, Callable, Optional

import pandas as pd

from ml_toolkit.data_splitting.data_splitter import DataSplitter
from ml_toolkit.training_manager.evaluator import Evaluator

class TrainingManager:
    def __init__(
        self, 
        method_config: Dict[str, Dict], 
        scoring_config: Dict[str, Dict], 
        splitter: DataSplitter, 
        methods: List[str], 
        data_paths: List[Tuple[str, str]],
        results_dir: Optional[str] = None
    ):
        """
        Initializes the TrainingManager.

        Args:
            method_config (Dict[str, Dict]): Configuration of methods with default parameters.
            scoring_config (Dict[str, Dict]): Configuration of scoring metrics.
            splitter (DataSplitter): An instance of the DataSplitter class.
            methods (List[str]): List of methods to train on each dataset.
            data_paths (List[Tuple[str, str]]): List of tuples (data_path, table_name). 
                If table_name is None, it's assumed that the dataset is not SQL-based.
            results_dir (str): Directory where results will be stored. If None a timestamp will be used.
        """
        self.method_config = method_config
        self.scoring_config = scoring_config
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

        self.evaluator = Evaluator(
            method_config=self.method_config, scoring_config=self.scoring_config
        )

        self.__validate_methods()
        self.data_splits = self.__get_data_splits()
        self.configurations = self.__create_configurations()

    def __validate_methods(self):
        """Check all methods are included in the method_config"""
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

    def __get_data_splits(
        self
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Splits each dataset using the provided splitter and returns a dictionary mapping
        each dataset path to its respective train-test splits.

        Returns:
            Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
                A dictionary where the key is the data path, and the value is the 
                train-test split for the dataset at that path.
        """
        data_splits = {}
        for data_path, table_name in self.data_paths:
            X_train, X_test, y_train, y_test = self.splitter.split(
                data_path, table_name
                )
            data_splits[data_path] = (X_train, X_test, y_train, y_test)
        
        return data_splits

    def __create_configurations(self) -> List[Tuple[str, str]]:
        """
        Create configurations as a Cartesian product of methods and datasets.

        Returns:
            List[Tuple[str, str]]:
                A list of tuples where each tuple represents (data_path, method).
        """
        if all(isinstance(method, str) for method in self.methods):
            method_combinations = [(method,) for method in self.methods]
        else:
            method_combinations = zip(*self.methods)

        configurations = deque(itertools.product(self.data_paths, method_combinations))
        return configurations

    def __get_results_dir(self):
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        return f"{timestamp}_Results"

    def __get_configuration_dir(
        self, 
        method_name: str, 
        data_path: Tuple[str, str],
        results_dir: str
    ) -> str:
        """
        Create a meaningful directory name for each configuration.

        Args:
            method_name (str): The name of the model method being used.
            data_path (str): The path to the dataset.

        Returns:
            str: A directory path to store results for this configuration.
        """
        dataset_name = os.path.basename(data_path[0]).split(".")[0]
        config_dir = f"{method_name}_{dataset_name}"
        full_path = os.path.join(results_dir, config_dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        return full_path

    def run_configurations(self, workflow: Callable):
        """
        Run the user-defined workflow for each configuration.

        Args:
            workflow_function (Callable): A function that defines the workflow 
                for each configuration.
        """
        error_log = []
        if not self.results_dir:
            results_dir = self.__get_results_dir()
        else:
            results_dir = self.results_dir

        os.makedirs(results_dir, exist_ok=True)

        while self.configurations:
            data_path, method_names = self.configurations.popleft()
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

                config_dir = self.__get_configuration_dir(
                    "_".join(method_names), data_path, results_dir
                    )
                config_evaluator = self.evaluator.with_config(
                    output_dir=config_dir
                    )
                workflow(
                    config_evaluator, X_train, X_test, y_train, y_test, 
                    config_dir, method_names, **model_kwargs
                    )

            except Exception as e:
                error_message = f"Error for {method_names} on {data_path}: {str(e)}"
                print(error_message)
                traceback.print_exc()
                error_log.append(error_message)
        
        if error_log:
            error_log_path = os.path.join(results_dir, "error_log.txt")
            with open(error_log_path, "w") as file:
                file.write("\n".join(error_log))
                