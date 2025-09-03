import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import abc

from brisk.services import base
from brisk.version import __version__
from brisk.configuration import project

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
        config = algorithm_config.export_params()
        self.rerun_service.add_algorithm_config(config)
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
        self.rerun_service.add_workflow_file(workflow_name)
        return workflow
 

class CoordinatingStrategy(RerunStrategy):
    """Strategy for coordinating mode - provides data from config file."""
    
    def __init__(self, config_data: Dict[str, Any]):
        self.config_data = config_data
        self._reconstructed_objects = {}
    
    def handle_load_base_data_manager(self, data_manager) -> Any:
        """Provide base data manager from config instead of loading from file."""
        # TODO: This method must the base DataManager insance from config file
        raise NotImplementedError(
            "handle_load_base_data_manager not implemented for CoordinatingStrategy"
        )
    
    def handle_load_algorithms(self, algorithm_config) -> Any:
        """Provide algorithms from config instead of loading from file."""
        # TODO: This method must the AlgorithmCollection insance from config file
        raise NotImplementedError(
            "handle_load_algorithms not implemented for CoordinatingStrategy"
        )
    
    
    def handle_load_custom_evaluators(self, module, evaluators_file: Path) -> Any:
        """Provide custom evaluators from config instead of loading from file."""
        raise NotImplementedError(
            "handle_load_custom_evaluators not implemented for CoordinatingStrategy"
        )
    
    def handle_load_workflow(self, workflow, workflow_name: str) -> Any:
        """Provide workflow from config instead of loading from file."""
        # TODO: this must return a Workflow instance 

        raise NotImplementedError(
            "handle_load_workflow not implemented for CoordinatingStrategy"
        )


class RerunService(base.BaseService):
    """
    Collects run-time configs (e.g., DataManager params) for exact reruns.

    - Stores configs in memory during the run.
    - Uses IOService.save_to_json(...) to write a single run_config.json at the end.
    """

    def __init__(self, name: str, mode: str = "capture"):
        super().__init__(name)
        self.configs: Dict[str, Any] = {
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
        elif mode == "coordianate":
            self.strategy = CoordinatingStrategy(self.configs)
            self.is_coordinating = True
        else:
            raise ValueError(
                f"Unkown mode: {mode}. Must be 'capture' or 'coordianate'"
            )

        self.mode = mode

    def add_base_data_manager(self, config: Dict[str, Any]) -> None:
        self.configs["base_data_manager"] = config
    
    def add_configuration(self, configuration: Dict[str, Any], groups: list[Dict[str, Any]]) -> None:
        self.configs["configuration"] = configuration
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

    def add_workflow_file(self, workflow_name: str):
        project_root = project.find_project_root()
        workflow_file = project_root / "workflows" / f"{workflow_name}.py"
        if workflow_file.exists():
            try:
                with open(workflow_file, "r", encoding="utf-8") as f:
                    file_content = f.read()

                self.configs["workflows"][f"{workflow_name}.py"] = file_content

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

    def set_configs(self, configs: Dict[str, Any]):
        self.configs = configs

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
        return self.strategy.handle_load_algorithms(algorithm_config)
    
    def handle_load_custom_evaluators(self, module, evaluators_file: Path) -> Any:
        """Delegate to current strategy."""
        return self.strategy.handle_load_custom_evaluators(
            module, evaluators_file
        )
    
    def handle_load_workflow(self, workflow, workflow_name: str) -> Any:
        """Delegate to current strategy."""
        return self.strategy.handle_load_workflow(workflow, workflow_name)
