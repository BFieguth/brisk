"""Manager for evaluating models and generating plots.

This module provides the EvaluationManager class, which coordinates evaluation
operations and manages services and evaluators for model evaluation and
visualization. Services implement functionality shared by all evaluators,
while evaluators implement specific evaluation methods.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import copy
import os

import numpy as np
from sklearn import base
import plotnine as pn

from brisk.adapters.filesystem.serializer_adapter import JoblibSerializerAdapter
from brisk.evaluation.evaluators import registry
from brisk.evaluation.evaluators import builtin
from brisk.ports import metric
from brisk.services import (
    get_services,
    update_experiment_config,
    missing,
    bundle
)
from brisk.evaluation.evaluators import base as base_eval
from brisk.configuration import project

_serializer = JoblibSerializerAdapter()


class EvaluationManager:
    """Coordinator for evaluation operations.

    Manages services and evaluators for model evaluation and visualization.
    Services implement functionality shared by all evaluators. Evaluators
    implement a specific evaluation method. EvaluationManager coordinates the
    use of services and evaluators.

    Parameters
    ----------
    metric_manager : MetricManager
        The metric configuration manager for handling evaluation metrics

    Attributes
    ----------
    services : ServiceBundle
        The global services bundle providing shared functionality
    metric_manager : MetricManager
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
        >>> from brisk import MetricManager
        >>> metric_mgr = MetricManager()
        >>> eval_mgr = EvaluationManager(metric_mgr)

    Get an evaluator:
        >>> evaluator = eval_mgr.get_evaluator("classification_metrics")
    """

    def __init__(
        self,
        metric_manager: metric.MetricManagerPort,
    ):
        """Initialize EvaluationManager with metric configuration.

        Parameters
        ----------
        metric_manager : MetricManager
            The metric configuration manager for handling evaluation metrics
        """
        self.services = missing.MissingServices()
        self.metric_manager = copy.deepcopy(metric_manager)
        self.output_dir = None
        self.plot_settings = None
        self.registry = registry.EvaluatorRegistry()

    def set_services(
        self,
        services: Optional[bundle.ServiceBundle] = None
    ) -> None:
        if services is None:
            self.services = get_services()
        else:
            self.services = services
        self.plot_settings = self.services.utility.get_plot_settings()

    def set_output_dir(self, output_dir: str) -> None:
        """
        Parameters
        ----------
        output_dir : str
            The output directory for the evaluation results

        Returns
        -------
        None
        """
        self.output_dir = Path(output_dir)

    def set_experiment_config(
        self,
        group_index_train: Dict[str, np.array],
        group_index_test: Dict[str, np.array],
        update_method: Optional[Callable] = None
    ) -> None:
        """Update services and metric_manager with the values for the current
        experiment.

        Parameters
        ----------
        group_index_train : Dict[str, np.array]
            The group index for the training split
        group_index_test : Dict[str, np.array]
            The group index for the testing split
        update_method: Optional[Callable]
            Inject a different update function; used for testing 

        Returns
        -------
        None

        Notes
        -----
        This method must be called before running evaluations to ensure
        that all services and metrics are properly configured for the
        current experiment context.
        """
        if update_method:
            update_method(
                self.output_dir, group_index_train, group_index_test
            )
        else:
            update_experiment_config(
                self.output_dir, group_index_train, group_index_test
            )

    def set_evaluator_registry(self):
        self.services.reporting.set_evaluator_registry(self.registry)

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
        self.metric_manager.set_split_metadata(split_metadata)

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

        # Custom evaluators are optional - skip if file doesn't exist
        if not evaluators_file.exists():
            return

        module = self.services.io.load_custom_evaluators(evaluators_file)
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

    def initialize_evaluators(self) -> None:
        """Initialize all evaluators with shared services.

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
        builtin.register_builtin_evaluators(self.registry, self.plot_settings)
        self._register_custom_evaluators(self.plot_settings)

    def set_evaluator_services(self) -> None:
        for evaluator in self.registry.evaluators.values():
            evaluator.set_services(self.services)

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
        evaluator.set_services(self.services)
        evaluator.set_metric_config(self.metric_manager)
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
        _serializer.dump(model_package, output_path)
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
        return _serializer.load(filepath)
