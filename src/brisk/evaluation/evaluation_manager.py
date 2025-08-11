"""Manager for evaluating models and generating plots.

Mananges services and evaluators for model evaluation and visualization.
Services implement functionality shared by all evaluators. Evaluators implement
a specific evaluation method. EvaluationManager coordinates the use of services
and evaluators.
"""

from pathlib import Path
from typing import Dict, Any
import copy
import os

import numpy as np
from sklearn import base
import joblib

from brisk.evaluation.evaluators import registry
from brisk.evaluation import metric_manager
from brisk.evaluation.evaluators import builtin
from brisk.services import get_services, update_experiment_config
from brisk.evaluation.evaluators.base import BaseEvaluator

class EvaluationManager:
    """Coordinator for evaluation operations.

    Manages services and evaluators for model evaluation and visualization.
    Services implement functionality shared by all evaluators. Evaluators
    implement a specific evaluation method. EvaluationManager coordinates the
    use of services and evaluators.

    Attributes
    ----------
    services : ServiceBundle
        The global services bundle
    metric_config : MetricManager
        The metric configuration manager
    output_dir : Path
        The output directory for the evaluation results
    registry : EvaluatorRegistry
        The evaluator registry with evaluators for models
    """
    def __init__(
        self,
        metric_config: metric_manager.MetricManager,
    ):
        self.services = get_services()
        self.metric_config = copy.deepcopy(metric_config)
        self.output_dir = None
        self.registry = registry.EvaluatorRegistry()
        self._initialize_builtin_evaluators()

    def set_experiment_values(
        self,
        output_dir: str,
        split_metadata: Dict[str, Any],
        group_index_train: Dict[str, np.array],
        group_index_test: Dict[str, np.array],
    ) -> None:
        """Update services and metric_config with the values for the current
        experiment.

        Parameters
        ----------
        output_dir : str
            The output directory for the evaluation results
        split_metadata : Dict[str, Any]
            The split metadata for the current experiment
        group_index_train : Dict[str, np.array]
            The group index for the training split
        group_index_test : Dict[str, np.array]
            The group index for the testing split

        Returns
        -------
        None
        """
        self.output_dir = Path(output_dir)
        update_experiment_config(
            self.output_dir, group_index_train, group_index_test
        )
        self.update_metrics(split_metadata)

    def update_metrics(self, split_metadata: Dict[str, Any]) -> None:
        """Update the metric configuration with the split metadata.

        Parameters
        ----------
        split_metadata : Dict[str, Any]
            The split metadata for the current experiment

        Returns
        -------
        None
        """
        self.metric_config.set_split_metadata(split_metadata)

    def _initialize_builtin_evaluators(self):
        """Initalize all built-in evaluators with shared services.

        This method registers all built-in evaluators with the evaluator 
        registry and sets the services for each evaluator.

        Returns
        -------
        None
        """
        builtin.register_builtin_evaluators(self.registry)

        for evaluator in self.registry.evaluators.values():
            evaluator.set_services(self.services)

        self.services.reporting.set_evaluator_registry(self.registry)

    def get_evaluator(self, name: str) -> BaseEvaluator:
        """Return an evaluator instance.

        Parameters
        ----------
        name : str
            The name of the evaluator

        Returns
        -------
        BaseEvaluator
            An evaluator instance
        """
        evaluator = self.registry.get(name)
        evaluator.set_metric_config(self.metric_config)
        return evaluator

    def save_model(self, model: base.BaseEstimator, filename: str) -> None:
        """Save model to pickle file.

        Parameters
        ----------
        model (BaseEstimator): 
            The model to save.

        filename (str): 
            The name for the output file (without extension).
        """
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.pkl")
        metadata = self.services.metadata.get_model(
            model, method_name="save_model"
        )
        model_package = {
            "model": model,
            "metadata": metadata
        }
        joblib.dump(model_package, output_path)
        self.services.logger.logger.info(
            "Saving model '%s' to '%s'.", filename, output_path
        )

    def load_model(self, filepath: str) -> base.BaseEstimator:
        """Load model from pickle file.

        Parameters
        ----------
        filepath : str
            Path to saved model file

        Returns
        -------
        BaseEstimator
            Loaded model

        Raises
        ------
        FileNotFoundError
            If model file does not exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        return joblib.load(filepath)
