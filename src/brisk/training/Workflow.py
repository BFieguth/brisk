"""The Workflow base class for defining training and evaluation steps.

Specific workflows (e.g., regression, classification) should 
inherit from this class and implement the abstract `workflow` method to define 
the steps for model training, evaluation, and reporting.

Exports:
    - Workflow: A base class for building machine learning workflows.
"""

import abc
from typing import List, Dict, Any

import pandas as pd

from brisk.evaluation.EvaluationManager import EvaluationManager

class Workflow:
    """Base class for machine learning workflows.

    Attributes:
        evaluator (Evaluator): An object for evaluating models.
        X_train (pd.DataFrame): The training feature data.
        X_test (pd.DataFrame): The test feature data.
        y_train (pd.Series): The training target data.
        y_test (pd.Series): The test target data.
        output_dir (str): The directory where results are saved.
        method_names (List[str]): Names of the methods/models used.
        model1, model2, ...: The models passed to the workflow.
    """
    def __init__(
        self,
        evaluator: EvaluationManager, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series, 
        output_dir: str, 
        method_names: List[str], 
        model_kwargs: Dict[str, Any],
        workflow_config = None
    ):
        self.evaluator = evaluator
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.output_dir = output_dir
        self.method_names = method_names
        self._unpack_attributes(model_kwargs)
        if workflow_config:
            self._unpack_attributes(workflow_config)

    def __getattr__(self, name: str) -> None:
        available_attrs = ", ".join(self.__dict__.keys())
        raise AttributeError(
            f"'{name}' not found. Available attributes are: {available_attrs}"
            )

    def _unpack_attributes(self, config: Dict[str, Any]) -> None:
        """
        Unpacks the key-value pairs from the config dictionary and sets them as 
        attributes of the instance.

        Args:
            config (Dict[str, Any]): The configuration dictionary to unpack.
        """
        for key, model in config.items():
            setattr(self, key, model)
 
    @abc.abstractmethod
    def workflow(self) -> None:
        raise NotImplementedError("Subclass must implement the workflow method.")
    