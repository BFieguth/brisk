"""Manager for evaluating models and generating plots.

This module provides the EvaluationManager class, which coordinates evaluation
operations and manages services and evaluators for model evaluation and
visualization. Services implement functionality shared by all evaluators,
while evaluators implement specific evaluation methods.
"""
from pathlib import Path
from typing import Dict, Any
import copy
import os

import numpy as np
from sklearn import base
import joblib
import plotnine as pn

from brisk.evaluation.evaluators import registry
from brisk.evaluation import metric_manager
from brisk.evaluation.evaluators import builtin
from brisk.services import get_services, update_experiment_config
from brisk.evaluation.evaluators import base as base_eval
from brisk.configuration import project

class EvaluationManager:
    """Coordinator for evaluation operations.

    Manages services and evaluators for model evaluation and visualization.
    Services implement functionality shared by all evaluators. Evaluators
    implement a specific evaluation method. EvaluationManager coordinates the
    use of services and evaluators.

    Parameters
    ----------
    metric_config : MetricManager
        The metric configuration manager for handling evaluation metrics

    Attributes
    ----------
    services : ServiceBundle
        The global services bundle providing shared functionality
    metric_config : MetricManager
        The metric configuration manager
    output_dir : Path or None
        The output directory for the evaluation results
    registry : EvaluatorRegistry
        The evaluator registry with evaluators for models

    Notes
    -----
    The EvaluationManager serves as the central coordinator for all evaluation
    operations. It initializes built-in and custom evaluators, provides access
    to the services bundle, and provides methods for model saving/loading and
    evaluator retrieval.

    Examples
    --------
    Initialize evaluation manager:
        >>> from brisk.evaluation import metric_manager
        >>> metric_mgr = metric_manager.MetricManager()
        >>> eval_mgr = EvaluationManager(metric_mgr)

    Get an evaluator:
        >>> evaluator = eval_mgr.get_evaluator("classification_metrics")
    """

    def __init__(
        self,
        metric_config: metric_manager.MetricManager,
    ):
        """Initialize EvaluationManager with metric configuration.

        Parameters
        ----------
        metric_config : MetricManager
            The metric configuration manager for handling evaluation metrics
        """
        self.services = get_services()
        self.metric_config = copy.deepcopy(metric_config)
        self.output_dir = None
        self.registry = registry.EvaluatorRegistry()
        self._initialize_evaluators()

    def set_experiment_values(
        self,
        output_dir: str,
        split_metadata: Dict[str, Any],
        group_index_train: Dict[str, np.array],
        group_index_test: Dict[str, np.array],
    ) -> None:
        """Update services and metric_config with the values for the current
        experiment.

        Sets up the evaluation environment for a specific experiment by
        configuring the output directory, updating experiment configuration,
        and setting split metadata for metrics.

        Parameters
        ----------
        output_dir : str
            The output directory for the evaluation results
        split_metadata : Dict[str, Any]
            The split metadata for the current experiment containing
            information about data splits
        group_index_train : Dict[str, np.array]
            The group index for the training split
        group_index_test : Dict[str, np.array]
            The group index for the testing split

        Returns
        -------
        None

        Notes
        -----
        This method must be called before running evaluations to ensure
        that all services and metrics are properly configured for the
        current experiment context.
        """
        self.output_dir = Path(output_dir)
        update_experiment_config(
            self.output_dir, group_index_train, group_index_test
        )
        self.update_metrics(split_metadata)

    def update_metrics(self, split_metadata: Dict[str, Any]) -> None:
        """Update the metric configuration with the split metadata.

        Configures the metric manager with information about the current
        data splits to ensure proper metric calculation.

        Parameters
        ----------
        split_metadata : Dict[str, Any]
            The split metadata for the current experiment containing
            information about data splits

        Returns
        -------
        None

        Notes
        -----
        This method updates the internal metric configuration to reflect
        the current experiment's data split characteristics.
        """
        self.metric_config.set_split_metadata(split_metadata)

    def _register_custom_evaluators(self, theme: pn.theme) -> None:
        """Register any Evaluators defined in evaluators.py.

        Loads and registers custom evaluators from the project's evaluators.py
        file. Custom evaluators must be properly registered to integrate
        with the Brisk framework.

        Parameters
        ----------
        theme : pn.theme
            The plotnine theme to use for custom evaluators

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If evaluators.py file is not found in the project root

        Notes
        -----
        This method looks for an evaluators.py file in the project root
        directory and attempts to load and register any custom evaluator
        classes defined within it.
        """
        project_root = project.find_project_root()
        evaluators_file = project_root / "evaluators.py"

        if evaluators_file.exists():
            module = self.services.io.load_custom_evaluators(evaluators_file)
        else:
            raise FileNotFoundError(
                f"evaluators.py not found in {project_root}"
            )

        if module:
            module.register_custom_evaluators(self.registry, theme)
            self._check_unregistered_evaluators(module)

    def _check_unregistered_evaluators(self, module) -> None:
        """Check for unregistered evaluator classes in the module.

        Identifies any evaluator classes in the module that were not
        properly registered and logs a warning for each unregistered class.

        Parameters
        ----------
        module : module
            The loaded evaluators module to check

        Returns
        -------
        None

        Notes
        -----
        This method scans the module for classes that inherit from
        BaseEvaluator but were not registered during the registration
        process. Unregistered evaluators cannot be used by the framework.
        """
        module_classes = [
            obj for _, obj in module.__dict__.items()
            if isinstance(obj, type) and obj.__module__ == module.__name__
        ]
        evaluator_classes = [
            obj for obj in module_classes
            if issubclass(obj, base_eval.BaseEvaluator)
        ]
        for obj in evaluator_classes:
            is_registered = any(
                isinstance(evaluator, obj)
                for evaluator in self.registry.evaluators.values()
            )

            if not is_registered:
                self.services.logger.logger.warning(
                    f"Found unregistered evalautor class {obj.__name__} in "
                    "evaluators.py. Evaluators must be registered to integrate "
                    "with Brisk."
                )

    def _initialize_evaluators(self) -> None:
        """Initialize all built-in evaluators with shared services.

        This method registers all built-in evaluators with the evaluator 
        registry and sets the services for each evaluator. It also attempts
        to register any custom evaluators from the project.

        Returns
        -------
        None

        Notes
        -----
        The initialization process includes:
        1. Registering built-in evaluators with plot settings
        2. Attempting to register custom evaluators
        3. Setting services for all evaluators
        4. Configuring the reporting service with the evaluator registry
        """
        plot_settings = self.services.utility.get_plot_settings()
        builtin.register_builtin_evaluators(self.registry, plot_settings)
        self._register_custom_evaluators(plot_settings)

        for evaluator in self.registry.evaluators.values():
            evaluator.set_services(self.services)

        self.services.reporting.set_evaluator_registry(self.registry)

    def get_evaluator(self, name: str) -> base_eval.BaseEvaluator:
        """Return an evaluator instance.

        Retrieves an evaluator from the registry and configures it with
        the current metric configuration.

        Parameters
        ----------
        name : str
            The name of the evaluator to retrieve

        Returns
        -------
        BaseEvaluator
            An evaluator instance configured with the current metric settings

        Notes
        -----
        The returned evaluator is configured with the current metric
        configuration, making it ready for use in evaluation operations.
        """
        evaluator = self.registry.get(name)
        evaluator.set_metric_config(self.metric_config)
        return evaluator

    def save_model(self, model: base.BaseEstimator, filename: str) -> None:
        """Save model to pickle file.

        Saves a trained model along with its metadata to a pickle file
        in the current output directory.

        Parameters
        ----------
        model : BaseEstimator
            The trained model to save
        filename : str
            The name for the output file (without extension)

        Returns
        -------
        None

        Notes
        -----
        The saved model package includes both the model object and its
        metadata. The output directory is created if it doesn't exist.
        The file is saved with a .pkl extension.
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

        Loads a previously saved model from a pickle file. The loaded
        model package includes both the model and its metadata.

        Parameters
        ----------
        filepath : str
            Path to the saved model file

        Returns
        -------
        BaseEstimator
            The loaded model object

        Raises
        ------
        FileNotFoundError
            If the model file does not exist at the specified path

        Notes
        -----
        This method loads the complete model package that was saved using
        the save_model method, which includes both the model and metadata.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        return joblib.load(filepath)
