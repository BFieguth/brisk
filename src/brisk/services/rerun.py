import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from brisk.services import base
from brisk.version import __version__
from brisk.configuration import project

class RerunService(base.BaseService):
    """
    Collects run-time configs (e.g., DataManager params) for exact reruns.

    - Stores configs in memory during the run.
    - Uses IOService.save_to_json(...) to write a single run_config.json at the end.
    """

    def __init__(self, name: str):
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
        self.capture_environment()


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

    def collect_workflow_files(self, configuration_json: Dict[str, Any], groups_json: List[Dict[str, Any]]) -> None:
        """
        Collect all workflow files used in the configuration for rerun functionality.
        
        Captures the entire content of each workflow file used, including the default
        workflow and any workflow overrides in experiment groups.
        
        Parameters
        ----------
        configuration_json : Dict[str, Any]
            The main configuration dictionary
        groups_json : List[Dict[str, Any]]
            List of experiment group configurations
        """
        project_root = project.find_project_root()
        workflows_dir = project_root / "workflows"
        workflow_files = set()
        
        default_workflow = configuration_json.get("default_workflow")
        if default_workflow:
            workflow_files.add(default_workflow)
        
        for group in groups_json:
            workflow_name = group.get("workflow")
            if workflow_name:
                workflow_files.add(workflow_name)
        
        workflows_content = {}
        for workflow_name in workflow_files:
            workflow_file = workflows_dir / f"{workflow_name}.py"
            if workflow_file.exists():
                try:
                    with open(workflow_file, 'r', encoding='utf-8') as f:
                        workflows_content[f"{workflow_name}.py"] = f.read()
                except (IOError, OSError) as e:
                    self._other_services["logger"].logger.warning(
                        f"Failed to read workflow file {workflow_name}.py: {e}"
                    )
            else:
                self._other_services["logger"].logger.warning(
                    f"Workflow file {workflow_name}.py not found"
                )
        
        self.configs["workflows"] = workflows_content

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
