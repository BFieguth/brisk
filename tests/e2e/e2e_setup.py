"""Define objects that will be reused by all e2e testing projects.
"""
import inspect
import os
import pathlib
import shutil
import subprocess
import sys
from typing import List, Callable

import numpy as np
from sklearn import linear_model, svm, ensemble
from sklearn import metrics as sk_metrics

import brisk
from brisk.configuration import configuration_manager as conf_manager
from tests import result_structure as rs
from brisk.configuration import project

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
        brisk.MetricWrapper(
            name="f1_multiclass",
            func=sk_metrics.f1_score,
            display_name="F1 Score (Multiclass)",
            average="weighted"
        ),
        brisk.MetricWrapper(
            name="precision_multiclass",
            func=sk_metrics.precision_score,
            display_name="Precision (Multiclass)",
            average="micro"
        ),
        brisk.MetricWrapper(
            name="recall_multiclass",
            func=sk_metrics.recall_score,
            display_name="Recall (Multiclass)",
            average="macro"
        ),
    )
    return metrics


def algorithm_config():
    algorithms = brisk.AlgorithmCollection(
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
        brisk.AlgorithmWrapper(
            name="xtree",
            display_name="Extra Tree Regressor",
            algorithm_class=ensemble.ExtraTreesRegressor,
            default_params={"min_samples_split": 10},
            hyperparam_grid={
                "n_estimators": list(range(20, 160, 20)),
                "criterion": ["friedman_mse", "absolute_error", 
                              "poisson", "squared_error"],
                "max_depth": list(range(5, 25, 5)) + [None]
            }
        ),
        brisk.AlgorithmWrapper(
            name="linear_svc",
            display_name="Linear Support Vector Classification",
            algorithm_class=svm.LinearSVC,
            default_params={"max_iter": 10000},
            hyperparam_grid={
                "C": list(np.arange(1, 30, 0.5)), 
                "penalty": ["l1", "l2"],
            }
        )
    )
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
        self.project_dir = pathlib.Path(
            os.path.join(self.e2e_dir, self.project_name)
        )
        self.results_dir = pathlib.Path(
            os.path.join(self.project_dir, "results")
        )
        self.settings_path = pathlib.Path(
            os.path.join(self.project_dir, "settings.py")
        )
        if self.settings_path.exists():
            self.original_settings = self.settings_path.read_text()

    def setup(self):
        """Setup the test project with required datasets."""
        project.find_project_root.cache_clear()
        self._write_settings_file()

        datasets_dir = os.path.join(self.e2e_dir, "datasets")
        project_datasets = os.path.join(self.project_dir, "datasets")
        os.makedirs(project_datasets, exist_ok=True)

        for dataset in self.datasets:
            src = os.path.join(datasets_dir, dataset)
            dst = os.path.join(project_datasets, dataset)
            shutil.copy2(src, dst)

        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)

    def run(self):
        """Run the workflow using brisk CLI run command."""
        if not self.project_dir:
            raise RuntimeError("Project not set up. Call setup() first.")

        project_root = str(pathlib.Path(__file__).parent.parent.parent)
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        env = os.environ.copy()
        if current_pythonpath:
            env["PYTHONPATH"] = f"{project_root}:{current_pythonpath}"
        else:
            env["PYTHONPATH"] = project_root

        use_shell = False
        if sys.platform == "win32":
            use_shell = True
        result = subprocess.run(
            [
                "brisk", "run",
                "-w", self.workflow_name,
                "-n", self.test_name
            ],
            cwd=str(self.project_dir),
            capture_output=True,
            text=True,
            check=False,
            env=env,
            shell=use_shell
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

        project_datasets = os.path.join(self.project_dir, "datasets")
        if os.path.exists(project_datasets):
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
