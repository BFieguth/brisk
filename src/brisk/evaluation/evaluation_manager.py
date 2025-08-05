"""Manager for evaluating models and generating plots.

Mananges services and evaluators for model evaluation and visualization.
Services implement functionality shared by all evaluators. Evaluators implement
a specific evaluation method. EvaluationManager coordinates the use of services
and evaluators.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import copy
import os

import numpy as np
from sklearn import base
import joblib

from .services.metadata import MetadataService
from .services.io import IOService
from .services.utility import UtilityService
from .services.bundle import ServiceBundle
from .evaluators.registry import EvaluatorRegistry
from ..configuration.algorithm_wrapper import AlgorithmCollection
from ..evaluation.metric_manager import MetricManager
from .evaluators.builtin import register_builtin_evaluators

class EvaluationManager:
    """Coordinator for evaluation operations.

    Manages services and evaluators for model evaluation and visualization.
    Services implement functionality shared by all evaluators. Evaluators
    implement a specific evaluation method. EvaluationManager coordinates the
    use of services and evaluators.
    """
    def __init__(
        self,
        algorithm_config: AlgorithmCollection,
        metric_config: MetricManager,
        output_dir: str,
        split_metadata: Dict[str, Any],
        group_index_train: Dict[str, np.array] | None,
        group_index_test: Dict[str, np.array] | None,
        logger: Optional[logging.Logger] = None,
    ):
        self.algorithm_config = algorithm_config
        self.metric_config = copy.deepcopy(metric_config)
        self.metric_config.set_split_metadata(split_metadata)
        self.output_dir = Path(output_dir)
        self.logger = logger

        # Initalize services
        self.metadata_service = MetadataService(
            self.output_dir, self.logger, self.algorithm_config
        )
        self.io_service = IOService(self.output_dir, self.logger)
        self.utility_service = UtilityService(
            self.output_dir, self.logger, self.algorithm_config,
            group_index_train, group_index_test
        )

        # Evaluation registry
        self.registry = EvaluatorRegistry()
        self._initialize_builtin_evaluators()

    def _initialize_builtin_evaluators(self):
        """Initalize all built-in evaluators with shared services."""
        service_bundle = ServiceBundle(
            metadata=self.metadata_service,
            io=self.io_service,
            utility=self.utility_service,
            metric_config=self.metric_config,
            logger=self.logger
        )
        register_builtin_evaluators(self.registry, service_bundle)

    def get_evaluator(self, name: str):
        """Return an evaluator instance."""
        return self.registry.get(name)

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
        metadata = self.metadata_service.get_metadata(
            model, method_name="save_model"
        )
        model_package = {
            "model": model,
            "metadata": metadata
        }
        joblib.dump(model_package, output_path)
        self.logger.info(
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
