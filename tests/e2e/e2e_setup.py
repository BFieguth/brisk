"""Define objects that will be reused by all e2e testing projects.
"""
import inspect
import os
import pathlib
import shutil
import subprocess
from typing import List, Callable

import numpy as np
from sklearn import linear_model, svm

import brisk
from brisk.configuration import configuration_manager as conf_manager
from brisk.utility import result_structure as rs
from brisk.utility import utility

def huber_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    DELTA = 1
    loss = np.where(
        np.abs(y_true - y_pred) <= DELTA,
        0.5 * (y_true - y_pred)**2,
        DELTA * (np.abs(y_true - y_pred) - 0.5 * DELTA)
    )
    return np.mean(loss)


def fake_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_metadata: dict
) -> float:
    """Function to check that split_metadata is applied properly"""
    return np.mean(
        (y_true - y_pred) / 
        (split_metadata["num_features"] / split_metadata["num_samples"])
    )


def metric_config():
    metrics = brisk.MetricManager(
        *brisk.REGRESSION_METRICS,
        *brisk.CLASSIFICATION_METRICS,
        brisk.MetricWrapper(
            name="huber_loss",
            func=huber_loss,
            display_name="Huber Loss", 
        ),
        brisk.MetricWrapper(
            name="fake_metric",
            func=fake_metric,
            display_name="Fake Metric"
        ),
    )
    return metrics


def algorithm_config():
    algorithms = [
        *brisk.REGRESSION_ALGORITHMS,
        *brisk.CLASSIFICATION_ALGORITHMS,
        brisk.AlgorithmWrapper(
            name="linear2",
            display_name="Linear Regression (Second)",
            algorithm_class=linear_model.LinearRegression
        ),
        brisk.AlgorithmWrapper(
            name="svc2",
            display_name="SVC (Second)",
            algorithm_class=svm.SVC
        ),
    ]
    return algorithms


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
        datasets: List[str],
        create_configuration: Callable[[], conf_manager.ConfigurationManager]
    ):
        self.test_name = test_name
        self.project_name = project_name
        self.datasets = datasets
        self.workflow_name = workflow_name
        self.create_configuration = create_configuration
        self.e2e_dir = pathlib.Path(__file__).parent
        self.project_dir = self.e2e_dir / self.project_name
        self.results_dir = self.project_dir / "results"
        self.settings_path = self.project_dir / "settings.py"
        if self.settings_path.exists():
            self.original_settings = self.settings_path.read_text()

    def setup(self):
        """Setup the test project with required datasets."""
        utility.find_project_root.cache_clear()
        self._write_settings_file()

        datasets_dir = self.e2e_dir / "datasets"
        project_datasets = self.project_dir / "datasets"
        project_datasets.mkdir(exist_ok=True)

        for dataset in self.datasets:
            shutil.copy2(
                datasets_dir / dataset,
                project_datasets / dataset
            )

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
            text=True,
            check=False
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
        """Clean up results directory, datasets and reset settings.py."""
        if self.results_dir and self.results_dir.exists():
            shutil.rmtree(self.results_dir)

        project_datasets = self.project_dir / "datasets"
        if project_datasets.exists():
            shutil.rmtree(project_datasets)

        if self.settings_path and self.original_settings:
            self.settings_path.write_text(self.original_settings)

    def _write_settings_file(self):
        """
        Inject the create_configuration function into settings.py at runtime
        """
        source = inspect.getsource(self.create_configuration)
        source_lines = source.splitlines()
        function_body = source_lines[1:]

        if function_body:
            indentation = min(len(line) - len(line.lstrip())
                             for line in function_body if line.strip())
            function_body = [line[indentation:] for line in function_body]

        function_body = ["    " + line if line.strip() else line
                        for line in function_body]
        formatted_source = "\n".join(function_body)
        settings_content = f"""
from brisk.configuration.configuration import Configuration

def create_configuration():
{formatted_source}"""
        self.settings_path.write_text(settings_content)

    def assert_result_strucure(self):
        result_dir = self.results_dir / self.test_name
        assert result_dir.exists()

        os.chdir(self.project_dir)
        config_manager = self.create_configuration()
        workflow_path = (
            self.project_dir / "workflows" / f"{self.workflow_name}.py"
        )

        expected_structure = rs.ResultStructure.from_config(
            config_manager, workflow_path
        )
        actual_structure = rs.ResultStructure.from_directory(result_dir)

        assert expected_structure == actual_structure
