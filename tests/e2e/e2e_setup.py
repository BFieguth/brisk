"""Define objects that will be reused by all e2e testing projects.
"""
import pathlib
import shutil
import subprocess
from typing import List, Optional

import brisk

def metric_config():
    METRIC_CONFIG = brisk.MetricManager(
        *brisk.REGRESSION_METRICS,
        *brisk.CLASSIFICATION_METRICS
    )
    return METRIC_CONFIG


def algorithm_config():
    ALGORITHM_CONFIG = [
        *brisk.REGRESSION_ALGORITHMS,
        *brisk.CLASSIFICATION_ALGORITHMS
    ]
    return ALGORITHM_CONFIG


class BaseE2ETest:
    """Base class for E2E testing of brisk projects.

    Args:
        project_name: Name of the project directory

        datasets: List of dataset filenames to copy from tests/e2e/datasets
        
        workflow_name: Name of the workflow file (without .py extension)
    """
    def __init__(
        self,
        test_name: str,
        project_name: str,
        workflow_name: str,
        datasets: List[str]
    ):
        self.test_name = test_name
        self.project_name = project_name
        self.datasets = datasets
        self.workflow_name = workflow_name
        self.project_dir: Optional[pathlib.Path] = None
        self.results_dir: Optional[pathlib.Path] = None

    def setup(self):
        """Setup the test project with required datasets."""
        e2e_dir = pathlib.Path(__file__).parent
        datasets_dir = e2e_dir / "datasets"
        self.project_dir = e2e_dir / self.project_name

        project_datasets = self.project_dir / "datasets"
        project_datasets.mkdir(exist_ok=True)

        for dataset in self.datasets:
            shutil.copy2(
                datasets_dir / dataset,
                project_datasets / dataset
            )
        
        self.results_dir = self.project_dir / "results"
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)

    def run(self):
        """Run the workflow using brisk CLI run command."""
        if not self.project_dir:
            raise RuntimeError("Project not set up. Call setup() first.")
        
        result = subprocess.run(
            [
                "brisk", "run",
                "-w", self.workflow_name,
                "-n", self.test_name
            ],
            cwd=str(self.project_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Workflow failed with error:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        
        self.stdout = result.stdout
        self.stderr = result.stderr

    def cleanup(self):
        """Clean up results directory."""
        if self.results_dir and self.results_dir.exists():
            shutil.rmtree(self.results_dir)
        
        project_datasets = self.project_dir / "datasets"
        if project_datasets.exists():
            shutil.rmtree(project_datasets)
