import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import abc
import tempfile
import os
import importlib
import inspect

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


class RerunStrategy(abc.ABC):
    """Abstract base class for rerun strategy."""

    @abc.abstractmethod
    def handle_load_base_data_manager(self, data_manager) -> Any:
        """Handle loading base data manager."""
        pass

    @abc.abstractmethod
    def handle_load_algorithms(self, algorithm_config) -> Any:
        """Handle loading algorithms."""
        pass
    
    @abc.abstractmethod
    def handle_load_custom_evaluators(self, module, evaluators_file: Path) -> Any:
        """Handle loading custom evaluators."""
        pass
    
    @abc.abstractmethod
    def handle_load_workflow(self, workflow, workflow_name: str) -> Any:
        """Handle loading workflow."""
        pass

    @abc.abstractmethod
    def handle_load_metric_config(self, metric_config) -> Any:
        """Handle loading metric config."""
        pass


class CaptureStrategy(RerunStrategy):
    """Strategy for capture mode - store data for config file."""

    def __init__(self, rerun_service):
        self.rerun_service = rerun_service
    
    def handle_load_base_data_manager(self, data_manager) -> Any:
        """Load data manager normally and capture its config.
        
        Parameters
        ----------
        data_manager: DataManager
            The DataManager instance loaded by the IOService from data.py
        """
        config = data_manager.export_params()
        self.rerun_service.add_base_data_manager(config)
        return data_manager
    
    def handle_load_algorithms(self, algorithm_config) -> Any:
        """Load algorithms normally and capture their config."""
        project_root = project.find_project_root()
        algorithms_file = project_root / "algorithms.py"
        
        if algorithms_file.exists():
            try:
                with open(algorithms_file, 'r', encoding='utf-8') as f:
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
    
    def handle_load_custom_evaluators(self, module, evaluators_file: Path) -> Any:
        """Load evaluators normally and capture their config.
        
        Captures the entire evaluators.py file content if it exists,
        since custom evaluators often have complex dependencies and
        user-defined classes that are best replicated as a complete file.
        """
        with open(evaluators_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        evaluator_config = {
            "type": "evaluators_file",
            "file_content": file_content,
            "file_path": "evaluators.py"
        }
        self.rerun_service.add_evaluators_config(evaluator_config)
        return module
    
    def handle_load_workflow(self, workflow, workflow_name: str) -> Any:
        """Load workflow normally and capture its content."""
        self.rerun_service.add_workflow_file(workflow_name, workflow.__name__)
        return workflow
 
    def handle_load_metric_config(self, metric_config) -> Any:
        """Load metric config normally and capture its content."""
        project_root = project.find_project_root()
        metrics_file = project_root / "metrics.py"
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
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
    """Strategy for coordinating mode - provides data from config file."""
    
    def __init__(self, config_data: Dict[str, Any]):
        self.config_data = config_data
        self._reconstructed_objects = {}
        self._temp_files = []
    
    def handle_load_base_data_manager(self, data_manager) -> Any:
        """Provide base data manager from config instead of loading from file."""
        preprocessor_classes = {
            'MissingDataPreprocessor': MissingDataPreprocessor,
            'ScalingPreprocessor': ScalingPreprocessor,
            'CategoricalEncodingPreprocessor': CategoricalEncodingPreprocessor,
            'FeatureSelectionPreprocessor': FeatureSelectionPreprocessor,
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
    
    def handle_load_algorithms(self, algorithm_config) -> Any:
        """Provide algorithms from config instead of loading from file."""
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
        
    def handle_load_custom_evaluators(self, module, evaluators_file: Path) -> Any:
        """Provide custom evaluators from config instead of loading from file."""
        raise NotImplementedError(
            "handle_load_custom_evaluators not implemented for CoordinatingStrategy"
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

    def _create_temp_file_from_content(self, file_content: str, filename: str) -> Path:
        """
        Helper method to create a temporary file from string content.
        
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
        """                

        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            prefix=filename.split(".")[0],
            suffix=".py", 
            delete=False,
            encoding='utf-8'
        )
        
        try:
            temp_file.write(file_content)
            temp_file.flush()
            temp_file_path = Path(temp_file.name)
            self._temp_files.append(temp_file_path)
            return temp_file_path
            
        finally:
            temp_file.close()

    def cleanup_temp_files(self):
        """Clean up all temporary files created during coordination."""        
        for temp_file_path in self._temp_files:
            try:
                if temp_file_path.exists():
                    os.unlink(temp_file_path)
            except OSError as e:
                print(f"Warning: Failed to cleanup temp file {temp_file_path}: {e}")

        self._temp_files.clear()

    def __del__(self):
        """Cleanup temporary files when strategy is destroyed."""
        self.cleanup_temp_files()


class RerunService(base.BaseService):
    """
    Collects run-time configs (e.g., DataManager params) for exact reruns.

    - Stores configs in memory during the run.
    - Uses IOService.save_to_json(...) to write a single run_config.json at the end.
    """

    def __init__(self, name: str, mode: str = "capture", rerun_config: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        self.configs: Dict[str, Any] = rerun_config or {
            "package_version": __version__,
            "created_at": datetime.now().isoformat(timespec="seconds"),
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
                f"Unkown mode: {mode}. Must be 'capture' or 'coordinate'"
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
            List of metric configurations exported from MetricManager.export_params()
        """
        self.configs["metrics"] = metric_configs

    def add_algorithm_config(self, algorithm_configs: List[Dict[str, Any]]) -> None:
        """
        Store algorithm configuration data for rerun functionality.
        
        Parameters
        ----------
        algorithm_configs : List[Dict[str, Any]]
            List of algorithm configurations exported from AlgorithmCollection.export_params()
        """
        self.configs["algorithms"] = algorithm_configs

    def add_evaluators_config(self, evaluators_config: Optional[Dict[str, Any]]) -> None:
        """
        Store evaluators configuration data for rerun functionality.
        
        Parameters
        ----------
        evaluators_config : Optional[Dict[str, Any]]
            Evaluators configuration exported from EvaluationManager.export_evaluators_config()
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

    def collect_dataset_metadata(self, groups_json: List[Dict[str, Any]]) -> None:
        """
        Collect metadata about all datasets used in experiment groups for rerun functionality.
        
        Captures dataset metadata including filename, table name, file size, and feature names
        to verify dataset compatibility during rerun.
        
        Parameters
        ----------
        groups_json : List[Dict[str, Any]]
            List of experiment group configurations containing dataset information
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

                    dataset_key_str = f"{dataset_name}|{table_name}" if table_name else dataset_name
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
                    f"Failed to collect metadata for dataset {dataset_name}: {e}"
                )
                dataset_metadata[dataset_key_str] = {
                    "filename": dataset_name,
                    "table_name": table_name,
                    "error": str(e)
                }
        
        self.configs["datasets"] = dataset_metadata

    def capture_environment(self) -> None:
        """Capture env info + pip freeze."""
        env = {"python": platform.python_version()}

        cp = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
        )
        env["pip_freeze"] = cp.stdout.strip() if cp.returncode == 0 and cp.stdout else None

        self.configs["env"] = env

    def export_and_save(self, results_dir: Path) -> None:
        """Write the run configuration to results/run_config.json."""
        out_path = Path(results_dir) / "run_config.json"
        # meta = {"kind": "brisk_rerun_config"} # TODO implelment w/ metadata service
        self._other_services["io"].save_rerun_config(data=self.configs, output_path=out_path)

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
    
    def handle_load_custom_evaluators(self, module, evaluators_file: Path) -> Any:
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
        self.get_service("reporting").set_metric_config(metric_config)
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
            "plot_settings": self.reconstruct_plot_settings(configuration["plot_settings"]),
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

    def reconstruct_plot_settings(self, plot_settings_data: Dict[str, Any]) -> 'PlotSettings':
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
                format=file_io_settings.get("format"),
                width=file_io_settings.get("width"),
                height=file_io_settings.get("height"),
                dpi=file_io_settings.get("dpi"),
                transparent=file_io_settings.get("transparent"),
                primary_color=colors.get("primary_color"),
                secondary_color=colors.get("secondary_color"),
                accent_color=colors.get("accent_color")
            )
            
            return plot_settings
            
        except Exception as e:
            self._other_services["logging"].logger.warning(
                f"Failed to reconstruct PlotSettings: {e}. Using defaults."
            )
            # return PlotSettings()
