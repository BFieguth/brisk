import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from brisk.services import base
from brisk.version import __version__

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
        }
        self.capture_environment()  # capture at initialization


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
        meta = {"kind": "brisk_rerun_config"}
        self.get_service("io").save_to_json(data=self.configs, output_path=out_path, metadata=meta)
