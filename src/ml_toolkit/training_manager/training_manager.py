import itertools
from typing import List, Dict, Tuple, Callable

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
        data_paths: List[Tuple[str, str]]
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
        """
        self.method_config = method_config
        self.scoring_config = scoring_config
        self.splitter = splitter
        self.methods = methods
        self.data_paths = data_paths

        self.evaluator = Evaluator(
            method_config=self.method_config, scoring_config=self.scoring_config
        )

        self.__validate_methods()
        self.data_splits = self.__get_data_splits()
        self.configurations = self.__create_configurations()

    def __validate_methods(self):
        """Check all methods are included in the method_config"""
        included_methods = self.method_config.keys()
        if set(self.methods).issubset(included_methods):
            return True
        else:
            invalid_methods = list(set(self.methods) - set(included_methods))
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
        configurations = list(itertools.product(self.data_paths, self.methods))
        return configurations
    
    def run_configurations(self, workflow: Callable):
        """
        Run the user-defined workflow for each configuration.

        Args:
            workflow_function (Callable): A function that defines the workflow 
                for each configuration.
        """
        for (data_path, method_name) in self.configurations:
            X_train, X_test, y_train, y_test = self.data_splits[data_path[0]]

            model = self.method_config[method_name].instantiate()
            
            workflow(self.evaluator, model, X_train, X_test, y_train, y_test) 
