"""Rerun service for capturing and coordinating experiment configurations.

This module provides comprehensive rerun functionality for the Brisk package,
enabling exact reproduction of machine learning experiments by capturing
runtime configurations and providing mechanisms to restore them. It implements
the Strategy pattern to handle different modes of operation: capture and
coordinate.

The rerun service supports capturing complete experiment configurations
including data managers, algorithms, evaluators, workflows, metrics, and
environment information. It can then coordinate the exact reproduction of these
experiments by reconstructing all necessary components from the captured
configuration.

Examples
--------
>>> from brisk.services.rerun import RerunService
>>> 
>>> # Capture mode - collect configuration during experiment
>>> rerun_service = RerunService("rerun", mode="capture")
>>> # ... run experiments ...
>>> rerun_service.export_and_save(results_dir)
>>> 
>>> # Coordinate mode - reproduce experiment from config
>>> with open("run_config.json") as f:
...     config_data = json.load(f)
>>> rerun_service = RerunService("rerun", mode="coordinate", rerun_config=config_data)
>>> # ... use captured configuration to reproduce experiments ...
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import abc
import tempfile
import os
import importlib
import inspect
import json

from brisk.services import base
from brisk.version import __version__
from brisk.configuration import project
from brisk.services import io
from brisk.configuration import algorithm_collection
from brisk.evaluation import metric_manager
from brisk.theme.plot_settings import PlotSettings
from brisk.theme.theme_serializer import ThemePickleJSONSerializer
from brisk.data import data_manager as data_manager_module
from brisk.data.preprocessing import (
    MissingDataPreprocessor,
    ScalingPreprocessor,
    CategoricalEncodingPreprocessor,
    FeatureSelectionPreprocessor
)
from brisk.cli.environment import EnvironmentManager


class RerunStrategy(abc.ABC):
    """Abstract base class for rerun strategy implementations.
    
    This abstract base class defines the interface for different rerun
    strategies used by the RerunService. It implements the Strategy pattern to
    allow different behaviors for capturing and coordinating experiment
    configurations.
    
    The strategy pattern enables the RerunService to switch between different
    modes of operation (capture vs coordinate) without changing the core service
    implementation.
    
    Notes
    -----
    All concrete strategy implementations must implement all abstract methods
    to handle the loading and processing of different experiment components.
    
    Examples
    --------
    >>> class CustomRerunStrategy(RerunStrategy):
    ...     def handle_load_base_data_manager(self, data_manager):
    ...         # Custom implementation
    ...         return data_manager
    ...     # ... implement other abstract methods
    """

    @abc.abstractmethod
    def handle_load_base_data_manager(self, data_manager: Any) -> Any:
        """Handle loading base data manager.
        
        Parameters
        ----------
        data_manager : Any
            The data manager instance to be processed
            
        Returns
        -------
        Any
            The processed data manager instance
        """
        pass

    @abc.abstractmethod
    def handle_load_algorithms(self, algorithm_config: Any) -> Any:
        """Handle loading algorithms.
        
        Parameters
        ----------
        algorithm_config : Any
            The algorithm configuration to be processed
            
        Returns
        -------
        Any
            The processed algorithm configuration
        """
        pass

    @abc.abstractmethod
    def handle_load_custom_evaluators(
        self,
        module: Any,
        evaluators_file: Path
    ) -> Any:
        """Handle loading custom evaluators.
        
        Parameters
        ----------
        module : Any
            The evaluators module to be processed
        evaluators_file : Path
            The path to the evaluators file
            
        Returns
        -------
        Any
            The processed evaluators module
        """
        pass

    @abc.abstractmethod
    def handle_load_workflow(self, workflow: Any, workflow_name: str) -> Any:
        """Handle loading workflow.
        
        Parameters
        ----------
        workflow : Any
            The workflow to be processed
        workflow_name : str
            The name of the workflow
            
        Returns
        -------
        Any
            The processed workflow
        """
        pass

    @abc.abstractmethod
    def handle_load_metric_config(self, metric_config: Any) -> Any:
        """Handle loading metric config.
        
        Parameters
        ----------
        metric_config : Any
            The metric configuration to be processed
            
        Returns
        -------
        Any
            The processed metric configuration
        """
        pass


class CaptureStrategy(RerunStrategy):
    """Strategy for capture mode - store data for config file.
    
    This strategy is used during experiment execution to capture and store
    all configuration data needed for exact reproduction. It loads components
    normally while simultaneously capturing their configuration parameters
    and file contents for later use in coordinate mode.
    
    The strategy captures:
    - Data manager parameters and preprocessor configurations
    - Algorithm configurations from algorithms.py files
    - Custom evaluators from evaluators.py files
    - Workflow files and class names
    - Metric configurations from metrics.py files
    
    Attributes
    ----------
    rerun_service : RerunService
        The rerun service instance for storing captured configurations
        
    Notes
    -----
    This strategy operates in "pass-through" mode, loading components normally
    while capturing their configuration data for later reproduction.
    
    Examples
    --------
    >>> rerun_service = RerunService("rerun", mode="capture")
    >>> strategy = CaptureStrategy(rerun_service)
    >>> data_manager = strategy.handle_load_base_data_manager(data_manager)
    """

    def __init__(self, rerun_service: "RerunService") -> None:
        """Initialize the capture strategy.
        
        Parameters
        ----------
        rerun_service : RerunService
            The rerun service instance for storing captured configurations
        """
        self.rerun_service = rerun_service

    def handle_load_base_data_manager(self, data_manager: Any) -> Any:
        """Load data manager normally and capture its config.
        
        This method loads the data manager normally while capturing its
        configuration parameters for later reproduction. The configuration
        includes all data manager parameters and preprocessor settings.
        
        Parameters
        ----------
        data_manager : Any
            The DataManager instance loaded by the IOService from data.py
            
        Returns
        -------
        Any
            The original data manager instance (pass-through behavior)
            
        Notes
        -----
        The captured configuration includes test_size, n_splits, split_method,
        group_column, stratified flag, random_state, and all preprocessor
        configurations.
        """
        config = data_manager.export_params()
        self.rerun_service.add_base_data_manager(config)
        return data_manager

    def handle_load_algorithms(self, algorithm_config: Any) -> Any:
        """Load algorithms normally and capture their config.
        
        This method loads the algorithm configuration normally while capturing
        the complete algorithms.py file content for later reproduction. The
        file content is stored as-is to preserve all algorithm definitions
        and configurations.
        
        Parameters
        ----------
        algorithm_config : Any
            The algorithm configuration loaded by the IOService
            
        Returns
        -------
        Any
            The original algorithm configuration (pass-through behavior)
            
        Notes
        -----
        The captured file content includes the entire algorithms.py file,
        preserving all algorithm definitions, hyperparameter grids, and
        any custom configurations that might be present.
        """
        project_root = project.find_project_root()
        algorithms_file = project_root / "algorithms.py"

        if algorithms_file.exists():
            try:
                with open(algorithms_file, "r", encoding="utf-8") as f:
                    file_content = f.read()

                algo_config = {
                    "type": "algorithms_file",
                    "file_content": file_content,
                    "file_path": "algorithms.py"
                }
                self.rerun_service.add_algorithm_config(algo_config)
            except (IOError, OSError) as e:
                self.rerun_service._other_services["logger"].logger.warning(
                    f"Failed to read algorithms.py file: {e}"
                )

        return algorithm_config

    def handle_load_custom_evaluators(
        self,
        module: Any,
        evaluators_file: Path
    ) -> Any:
        """Load evaluators normally and capture their config.
        
        This method loads the custom evaluators module normally while capturing
        the complete evaluators.py file content for later reproduction. Custom
        evaluators often have complex dependencies and user-defined classes
        that are best replicated as a complete file.
        
        Parameters
        ----------
        module : Any
            The evaluators module loaded by the IOService
        evaluators_file : Path
            The path to the evaluators.py file
            
        Returns
        -------
        Any
            The original evaluators module (pass-through behavior)
            
        Notes
        -----
        The captured file content includes the entire evaluators.py file,
        preserving all custom evaluator definitions, imports, and any
        complex dependencies that might be present.
        """
        with open(evaluators_file, "r", encoding="utf-8") as f:
            file_content = f.read()

        evaluator_config = {
            "type": "evaluators_file",
            "file_content": file_content,
            "file_path": "evaluators.py"
        }
        self.rerun_service.add_evaluators_config(evaluator_config)
        return module

    def handle_load_workflow(self, workflow: Any, workflow_name: str) -> Any:
        """Load workflow normally and capture its content.
        
        This method loads the workflow normally while capturing the workflow
        file content and class name for later reproduction. The workflow
        file is read from the workflows directory.
        
        Parameters
        ----------
        workflow : Any
            The workflow class loaded by the IOService
        workflow_name : str
            The name of the workflow file (without .py extension)
            
        Returns
        -------
        Any
            The original workflow class (pass-through behavior)
            
        Notes
        -----
        The captured workflow includes the complete file content and the
        class name, enabling exact reproduction of the workflow.
        """
        self.rerun_service.add_workflow_file(workflow_name, workflow.__name__)
        return workflow

    def handle_load_metric_config(self, metric_config: Any) -> Any:
        """Load metric config normally and capture its content.
        
        This method loads the metric configuration normally while capturing
        the complete metrics.py file content for later reproduction. The
        file content is stored as-is to preserve all metric definitions
        and configurations.
        
        Parameters
        ----------
        metric_config : Any
            The metric configuration loaded by the IOService
            
        Returns
        -------
        Any
            The original metric configuration (pass-through behavior)
            
        Notes
        -----
        The captured file content includes the entire metrics.py file,
        preserving all metric definitions, display names, and any
        custom configurations that might be present.
        """
        project_root = project.find_project_root()
        metrics_file = project_root / "metrics.py"

        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    file_content = f.read()

                config = {
                    "type": "metrics_file",
                    "file_content": file_content,
                    "file_path": "metrics.py"
                }
                self.rerun_service.add_metric_config(config)
            except (IOError, OSError) as e:
                self.rerun_service._other_services["logger"].logger.warning(
                    f"Failed to read metrics.py file: {e}"
                )

        return metric_config


class CoordinatingStrategy(RerunStrategy):
    """Strategy for coordinating mode - provides data from config file.
    
    This strategy is used during experiment reproduction to reconstruct
    all components from previously captured configuration data. It creates
    temporary files from captured content and reconstructs objects to
    enable exact reproduction of the original experiment.
    
    The strategy reconstructs:
    - Data managers with original parameters and preprocessors
    - Algorithm configurations from captured file content
    - Custom evaluators from captured file content
    - Workflow classes from captured file content
    - Metric configurations from captured file content
    
    Attributes
    ----------
    config_data : Dict[str, Any]
        The captured configuration data from the original experiment
    _reconstructed_objects : Dict[str, Any]
        Cache for reconstructed objects to avoid duplicate reconstruction
    _temp_files : List[Path]
        List of temporary files created during reconstruction
        
    Notes
    -----
    This strategy operates in "reconstruction" mode, creating temporary
    files from captured content and reconstructing objects to match
    the original experiment configuration exactly.
    
    Examples
    --------
    >>> with open("run_config.json") as f:
    ...     config_data = json.load(f)
    >>> strategy = CoordinatingStrategy(config_data)
    >>> data_manager = strategy.handle_load_base_data_manager(None)
    """

    def __init__(self, config_data: Dict[str, Any]) -> None:
        """Initialize the coordinating strategy.
        
        Parameters
        ----------
        config_data : Dict[str, Any]
            The captured configuration data from the original experiment
        """
        self.config_data = config_data
        self._reconstructed_objects = {}
        self._temp_files = []

    def handle_load_base_data_manager(self, data_manager: Any) -> Any:
        """Provide base data manager from config instead of loading from file.
        
        This method reconstructs a DataManager instance from the captured
        configuration data, including all original parameters and preprocessors.
        The reconstructed data manager will have identical behavior to the
        original one used in the experiment.
        
        Parameters
        ----------
        data_manager : Any
            Ignored parameter (maintained for interface compatibility)
            
        Returns
        -------
        data_manager_module.DataManager
            A reconstructed DataManager instance with original parameters
            and preprocessors
            
        Notes
        -----
        The reconstruction process:
        1. Extracts data manager parameters from captured config
        2. Reconstructs preprocessor instances with original parameters
        3. Creates new DataManager with original configuration
        
        Raises
        ------
        KeyError
            If required configuration data is missing
        ValueError
            If preprocessor class names are not recognized
        """
        preprocessor_classes = {
            "MissingDataPreprocessor": MissingDataPreprocessor,
            "ScalingPreprocessor": ScalingPreprocessor,
            "CategoricalEncodingPreprocessor": CategoricalEncodingPreprocessor,
            "FeatureSelectionPreprocessor": FeatureSelectionPreprocessor,
        }
        data_manager_config = self.config_data["base_data_manager"]["params"]
        preprocessor_config = data_manager_config.pop("preprocessors")
        preprocessors=[]
        for class_name, params in preprocessor_config.items():
            preprocessor_class = preprocessor_classes[class_name]
            preprocessor = preprocessor_class(**params)
            preprocessors.append(preprocessor)

        return data_manager_module.DataManager(
            **data_manager_config,
            preprocessors=preprocessors
        )

    def handle_load_algorithms(self, algorithm_config: Any) -> Any:
        """Provide algorithms from config instead of loading from file.
        
        This method reconstructs an AlgorithmCollection instance from the
        captured algorithms.py file content. It creates a temporary file
        with the original content and loads the configuration from it.
        
        Parameters
        ----------
        algorithm_config : Any
            Ignored parameter (maintained for interface compatibility)
            
        Returns
        -------
        algorithm_collection.AlgorithmCollection
            A reconstructed AlgorithmCollection instance with original
            algorithm definitions and configurations
            
        Raises
        ------
        ValueError
            If no algorithms configuration is found or if the loaded
            object is not a valid AlgorithmCollection
        FileNotFoundError
            If the temporary file cannot be created
        """
        if "algorithms" not in self.config_data:
            raise ValueError("No algorithms file found in rerun configuration")

        config = self.config_data["algorithms"]
        temp_file_path = self._create_temp_file_from_content(
            config["file_content"],
            config["file_path"]
        )

        algorithm_config_obj = io.IOService.load_module_object(
            str(temp_file_path.parent),
            temp_file_path.name,
            "ALGORITHM_CONFIG"
        )

        if not isinstance(
            algorithm_config_obj, algorithm_collection.AlgorithmCollection
        ):
            raise ValueError(
                "ALGORITHM_CONFIG is not a valid AlgorithmCollection instance"
            )

        self._reconstructed_objects["algorithms"] = algorithm_config_obj
        return self._reconstructed_objects["algorithms"]

    def handle_load_custom_evaluators(
        self,
        module: Any,
        evaluators_file: Path
    ) -> Any:
        """
        Provide custom evaluators from config instead of loading from file.
        """
        if "evaluators" not in self.config_data:
            raise ValueError("No evaluators file found in rerun configuration")

        config = self.config_data["evaluators"]
        temp_file_path = self._create_temp_file_from_content(
            config["file_content"],
            config["file_path"]
        )
        module_path = os.path.join(
            str(temp_file_path.parent), temp_file_path.name
        )
        if not os.path.exists(module_path):
            raise FileNotFoundError(
                f"{temp_file_path.name} not found in "
                f"{str(temp_file_path.parent)}"
            )
        module_name = os.path.splitext(temp_file_path.name)[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "register_custom_evaluators"):
            return module
        else:
            raise AttributeError(
                f"The object 'register_custom_evaluators' is not defined in "
                f"{temp_file_path.name}"
            )

    def handle_load_workflow(self, workflow, workflow_name: str) -> Any:
        """Provide workflow from config instead of loading from file."""
        if f"{workflow_name}.py" not in self.config_data["workflows"]:
            raise ValueError(
                f"Workflow {workflow_name}.py not found in rerun configuration"
            )
        workflow_config = self.config_data["workflows"][f"{workflow_name}.py"]
        temp_file_path = self._create_temp_file_from_content(
            workflow_config["file_content"],
            f"{workflow_name}.py"
        )

        spec = importlib.util.spec_from_file_location(
            f"temp_workflow_{workflow_name}",
            temp_file_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        workflow_classes = [
            obj for name, obj in inspect.getmembers(module)
            if name == workflow_config["class_name"]
        ]

        return workflow_classes[0]

    def handle_load_metric_config(self, metric_config) -> Any:
        """Provide metric config from config instead of loading from file."""
        if "metrics" not in self.config_data:
            raise ValueError("No metrics file found in rerun configuration")

        config = self.config_data["metrics"]
        temp_file_path = self._create_temp_file_from_content(
            config["file_content"],
            config["file_path"]
        )

        metric_config_obj = io.IOService.load_module_object(
            str(temp_file_path.parent),
            temp_file_path.name,
            "METRIC_CONFIG"
        )

        if not isinstance(
            metric_config_obj, metric_manager.MetricManager
        ):
            raise ValueError(
                "METRIC_CONFIG is not a valid MetricManager instance"
            )

        self._reconstructed_objects["metrics"] = metric_config_obj
        return self._reconstructed_objects["metrics"]

    def _create_temp_file_from_content(
        self,
        file_content: str,
        filename: str
    ) -> Path:
        """Helper method to create a temporary file from string content.
        
        This method creates a temporary Python file with the specified content
        and filename. The file is added to the cleanup list for later removal.
        
        Parameters
        ----------
        file_content : str
            The content to write to the file
        filename : str
            The desired filename (for reference and extension)
            
        Returns
        -------
        Path
            Path to the created temporary file
            
        Notes
        -----
        The temporary file is created with the .py extension and UTF-8 encoding.
        The file is added to the _temp_files list for cleanup when the strategy
        is destroyed or cleanup_temp_files() is called.
        """

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            prefix=filename.split(".")[0],
            suffix=".py",
            delete=False,
            encoding="utf-8"
        )

        try:
            temp_file.write(file_content)
            temp_file.flush()
            temp_file_path = Path(temp_file.name)
            self._temp_files.append(temp_file_path)
            return temp_file_path

        finally:
            temp_file.close()

    def cleanup_temp_files(self) -> None:
        """Clean up all temporary files created during coordination.
        
        This method removes all temporary files created during the
        reconstruction process. It should be called when the strategy
        is no longer needed to free up disk space.
        
        Notes
        -----
        The method attempts to remove each temporary file and logs warnings
        for any files that cannot be removed. The _temp_files list is
        cleared after cleanup.
        """
        for temp_file_path in self._temp_files:
            try:
                if temp_file_path.exists():
                    os.unlink(temp_file_path)
            except OSError as e:
                print(
                    f"Warning: Failed to cleanup temp file {temp_file_path}: "
                    f"{e}"
                )

        self._temp_files.clear()

    def __del__(self) -> None:
        """Cleanup temporary files when strategy is destroyed.
        
        This destructor ensures that temporary files are cleaned up
        when the strategy object is garbage collected.
        """
        self.cleanup_temp_files()


class RerunService(base.BaseService):
    """Main service class for managing rerun functionality.
    
    This service provides comprehensive rerun functionality for the Brisk
    package, enabling exact reproduction of machine learning experiments by
    capturing runtime configurations and providing mechanisms to restore them.
    It implements the Strategy pattern to handle different modes of operation:
    capture and coordinate.
    
    The service supports two modes:
    - Capture mode: Collects and stores all experiment configurations during
    execution
    - Coordinate mode: Reconstructs and provides configurations for exact
    reproduction
    
    Attributes
    ----------
    configs : Dict[str, Any]
        Dictionary storing all captured configuration data
    strategy : RerunStrategy
        The current strategy implementation
    mode : str
        The current mode of operation ("capture" or "coordinate")
    is_coordinating : bool
        Boolean flag indicating if the service is in coordinating mode
        
    Notes
    -----
    The service uses the Strategy pattern to switch between capture and
    coordinate modes without changing the core implementation. All configuration
    data is stored in memory and can be exported to a JSON file for later use.
    
    Examples
    --------
    >>> # Capture mode
    >>> rerun_service = RerunService("rerun", mode="capture")
    >>> # ... run experiments ...
    >>> rerun_service.export_and_save(results_dir)
    >>> 
    >>> # Coordinate mode
    >>> with open("run_config.json") as f:
    ...     config_data = json.load(f)
    >>> rerun_service = RerunService("rerun", mode="coordinate", rerun_config=config_data)
    """

    def __init__(
        self,
        name: str,
        mode: str = "capture",
        rerun_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the rerun service.
        
        This constructor sets up the rerun service with the specified mode
        and initializes the appropriate strategy. In capture mode, it starts
        with empty configuration data. In coordinate mode, it uses the
        provided configuration data.
        
        Parameters
        ----------
        name : str
            The name identifier for this service
        mode : str, default="capture"
            The mode of operation ("capture" or "coordinate")
        rerun_config : Optional[Dict[str, Any]], default=None
            Configuration data for coordinate mode (ignored in capture mode)
            
        Raises
        ------
        ValueError
            If mode is not "capture" or "coordinate"
        """
        super().__init__(name)
        self.configs: Dict[str, Any] = rerun_config or {
            "package_version": __version__,
            "env": {},
            "base_data_manager": None,
            "configuration": {},
            "experiment_groups": [],
            "metrics": [],
            "algorithms": [],
            "evaluators": None,
            "workflows": {},
            "datasets": {},
        }

        if mode == "capture":
            self.strategy = CaptureStrategy(self)
            self.capture_environment()
            self.is_coordinating = False
        elif mode == "coordinate":
            self.strategy = CoordinatingStrategy(self.configs)
            self.is_coordinating = True
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Must be 'capture' or 'coordinate'"
            )

        self.mode = mode

    def add_base_data_manager(self, config: Dict[str, Any]) -> None:
        self.configs["base_data_manager"] = config

    def add_configuration(self, configuration: Dict[str, Any]) -> None:
        self.configs["configuration"] = configuration

    def add_experiment_groups(self, groups: List[Dict[str, Any]]) -> None:
        self.configs["experiment_groups"] = groups

    def add_metric_config(self, metric_configs: List[Dict[str, Any]]) -> None:
        """
        Store metric configuration data for rerun functionality.
        
        Parameters
        ----------
        metric_configs : List[Dict[str, Any]]
            List of metric configurations exported from
            MetricManager.export_params()
        """
        self.configs["metrics"] = metric_configs

    def add_algorithm_config(
        self,
        algorithm_configs: List[Dict[str, Any]]
    ) -> None:
        """
        Store algorithm configuration data for rerun functionality.
        
        Parameters
        ----------
        algorithm_configs : List[Dict[str, Any]]
            List of algorithm configurations exported from
            AlgorithmCollection.export_params()
        """
        self.configs["algorithms"] = algorithm_configs

    def add_evaluators_config(
        self,
        evaluators_config: Optional[Dict[str, Any]]
    ) -> None:
        """
        Store evaluators configuration data for rerun functionality.
        
        Parameters
        ----------
        evaluators_config : Optional[Dict[str, Any]]
            Evaluators configuration exported from
            EvaluationManager.export_evaluators_config()
            Can be None if no custom evaluators exist
        """
        self.configs["evaluators"] = evaluators_config

    def add_workflow_file(self, workflow_name: str, class_name: str):
        project_root = project.find_project_root()
        workflow_file = project_root / "workflows" / f"{workflow_name}.py"
        if workflow_file.exists():
            try:
                with open(workflow_file, "r", encoding="utf-8") as f:
                    file_content = f.read()

                self.configs["workflows"][f"{workflow_name}.py"] = {
                    "file_content": file_content,
                    "class_name": class_name
                }

            except (IOError, OSError) as e:
                self._other_services["logger"].logger.warning(
                    f"Failed to read workflow file {workflow_name}.py: {e}"
                )
        else:
            self._other_services["logger"].logger.warning(
                f"Workflow file {workflow_name}.py not found"
            )

    def collect_dataset_metadata(
        self,
        groups_json: List[Dict[str, Any]]
    ) -> None:
        """
        Collect metadata about all datasets used in experiment groups for rerun
        functionality.
        
        Captures dataset metadata including filename, table name, file size, and
        feature names to verify dataset compatibility during rerun.
        
        Parameters
        ----------
        groups_json : List[Dict[str, Any]]
            List of experiment group configurations containing dataset
            information
        """
        project_root = project.find_project_root()
        datasets_dir = project_root / "datasets"
        dataset_metadata = {}

        unique_datasets = set()
        for group in groups_json:
            datasets = group.get("datasets", [])
            for dataset_info in datasets:
                dataset_name = dataset_info.get("dataset")
                table_name = dataset_info.get("table_name")
                dataset_key = (dataset_name, table_name)
                unique_datasets.add(dataset_key)

        for dataset_name, table_name in unique_datasets:
            try:
                dataset_path = datasets_dir / dataset_name
                if dataset_path.exists():

                    df = self.get_service("io").load_data(
                        str(dataset_path), table_name
                    )

                    feature_names = list(df.columns)
                    dataset_shape = df.shape

                    dataset_key_str = (
                        f"{dataset_name}|{table_name}"
                        if table_name
                        else dataset_name
                    )
                    dataset_metadata[dataset_key_str] = {
                        "filename": dataset_name,
                        "table_name": table_name,
                        "feature_names": feature_names,
                        "num_features": len(feature_names),
                        "num_samples": dataset_shape[0]
                    }
                else:
                    self._other_services["logger"].logger.warning(
                        f"Dataset file {dataset_name} not found"
                    )
                    dataset_metadata[dataset_key_str] = {
                        "filename": dataset_name,
                        "table_name": table_name,
                        "error": "File not found"
                    }

            except IOError as e:
                self._other_services["logger"].logger.warning(
                    f"Failed to collect metadata for dataset {dataset_name}: "
                    f"{e}"
                )
                dataset_metadata[dataset_key_str] = {
                    "filename": dataset_name,
                    "table_name": table_name,
                    "error": str(e)
                }

        self.configs["datasets"] = dataset_metadata

    def capture_environment(self) -> None:
        """Capture structured environment information."""       
        env_manager = EnvironmentManager(project.find_project_root())
        env = env_manager.capture_environment()
        self.configs["env"] = env

    def export_and_save(self, results_dir: Path) -> None:
        """Write the run configuration to results/run_config.json.
        
        This method exports all captured configuration data to a JSON file
        in the results directory. The file can be used later to reproduce
        the exact experiment configuration.
        
        Parameters
        ----------
        results_dir : Path
            The results directory where the configuration file will be saved
            
        Notes
        -----
        This method only operates in capture mode. In coordinate mode,
        the configuration data is already loaded from an existing file.
        The exported file contains all necessary information to reproduce
        the experiment including data managers, algorithms, evaluators,
        workflows, metrics, and environment information.
        """
        out_path = Path(results_dir) / "run_config.json"
        metadata = self.get_service("metadata").get_rerun("export_and_save")
        if self.mode == "capture":
            self._other_services["io"].save_rerun_config(
                data=self.configs, metadata=metadata, output_path=out_path
            )

    def handle_load_base_data_manager(self, data_manager) -> Any:
        """Delegate to current strategy.
        
        Parameters
        ----------
        data_manager: DataManager
            The DataManager instance loaded by the IOService from data.py
        """
        return self.strategy.handle_load_base_data_manager(data_manager)

    def handle_load_algorithms(self, algorithm_config: Path) -> Any:
        """Delegate to current strategy."""
        algo_config = self.strategy.handle_load_algorithms(algorithm_config)
        self.get_service("utility").set_algorithm_config(algo_config)
        self.get_service("metadata").set_algorithm_config(algo_config)
        return algo_config

    def handle_load_custom_evaluators(
        self,
        module,
        evaluators_file: Path
    ) -> Any:
        """Delegate to current strategy."""
        return self.strategy.handle_load_custom_evaluators(
            module, evaluators_file
        )

    def handle_load_workflow(self, workflow, workflow_name: str) -> Any:
        """Delegate to current strategy."""
        return self.strategy.handle_load_workflow(workflow, workflow_name)

    def handle_load_metric_config(self, metric_config) -> Any:
        """Delegate to current strategy."""
        config = self.strategy.handle_load_metric_config(metric_config)
        self.get_service("reporting").set_metric_config(config)
        return config

    def get_configuration_args(self) -> Dict:
        configuration = self.configs["configuration"]
        categorical_features = {}
        for group in configuration["categorical_features"]:
            if group["table_name"]:
                categorical_features[(group["dataset"], group["table_name"])] = group["features"]
            else:
                categorical_features[group["dataset"]] = group["features"]
        args = {
            "default_workflow": configuration["default_workflow"],
            "default_algorithms": configuration["default_algorithms"],
            "categorical_features": categorical_features,
            "default_workflow_args": configuration["default_workflow_args"],
            "plot_settings": self.reconstruct_plot_settings(
                configuration["plot_settings"]
            ),
        }
        return args

    def get_experiment_groups(self):
        experiment_groups = self.configs["experiment_groups"]
        for group in experiment_groups:
            for idx, dataset in enumerate(group["datasets"]):
                if dataset["table_name"]:
                    group["datasets"][idx] = (
                        dataset["dataset"], dataset["table_name"]
                    )
                else:
                    group["datasets"][idx] = dataset["dataset"]
        return experiment_groups

    def reconstruct_plot_settings(
        self,
        plot_settings_data: Dict[str, Any]
    ) -> "PlotSettings":
        """
        Reconstruct a PlotSettings instance from exported parameters.
        
        Parameters
        ----------
        plot_settings_data : dict
            Dictionary containing exported PlotSettings data
        
        Returns
        -------
        PlotSettings
            Reconstructed PlotSettings instance
        """
        if not plot_settings_data:
            return PlotSettings()

        try:
            file_io_settings = plot_settings_data.get("file_io_settings", {})
            colors = plot_settings_data.get("colors", {})
            theme_json = plot_settings_data.get("theme_json")

            theme = None
            if theme_json:
                serializer = ThemePickleJSONSerializer()
                theme = serializer.theme_from_json(theme_json)

            plot_settings = PlotSettings(
                theme=theme,
                override=True,
                file_format=file_io_settings.get("file_format"),
                width=file_io_settings.get("width"),
                height=file_io_settings.get("height"),
                dpi=file_io_settings.get("dpi"),
                transparent=file_io_settings.get("transparent"),
                primary_color=colors.get("primary_color"),
                secondary_color=colors.get("secondary_color"),
                accent_color=colors.get("accent_color")
            )

            return plot_settings

        except (json.JSONDecodeError, TypeError) as e:
            self._other_services["logging"].logger.warning(
                f"Failed to reconstruct PlotSettings: {e}. Using defaults."
            )
