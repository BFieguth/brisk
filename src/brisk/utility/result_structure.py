"""
This module contains the ResultStructure class, which is used to represent the
structure of the results directory of a Brisk experiment as booleans.
"""

import ast
import dataclasses
import json
import pathlib
from typing import Dict, Optional, Set, Tuple, List

import PIL

from brisk.configuration import configuration_manager, experiment_group

class MethodCallVisitor(ast.NodeVisitor):
    """Visitor that extracts method calls from an AST."""

    def __init__(self):
        self.method_calls = set()

    def visit_Call(self, node): # pylint: disable=C0103
        """Visit a call node in the AST."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == "self":
                    self.method_calls.add(node.func.attr)
        self.generic_visit(node)

    @classmethod
    def extract_workflow_methods(cls, workflow_path: str) -> Set[str]:
        """Extract all method calls from a workflow file."""
        with open(workflow_path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read())

        visitor = cls()

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if (
                        isinstance(item, ast.FunctionDef) and
                        item.name == "workflow"
                    ):
                        visitor.visit(item)

        return visitor.method_calls


@dataclasses.dataclass
class ConfigLog:
    file_exists: bool


@dataclasses.dataclass
class ErrorLog:
    """
    Flag to indicate if the error log file exists.
    """
    file_exists: bool


@dataclasses.dataclass
class HTMLReport:
    """
    Represents the structure of the HTML report directory.
    """
    index_exists: bool
    index_css_exists: bool
    experiment_css_exists: bool
    dataset_css_exists: bool
    dataset_pages: Dict[str, bool]
    experiment_pages: Dict[str, bool]


@dataclasses.dataclass
class ExperimentDirectory:
    """
    Represents the structure of an experiment directory.
    """
    save_model: bool
    evaluate_model: bool
    evaluate_model_cv: bool
    compare_models: bool
    plot_pred_vs_obs: bool
    plot_learning_curve: bool
    plot_feature_importance: bool
    plot_residuals: bool
    plot_model_comparison: bool
    confusion_matrix: bool
    plot_confusion_heatmap: bool
    plot_roc_curve: bool
    plot_precision_recall_curve: bool


@dataclasses.dataclass
class DatasetDirectory:
    """
    Represents the structure of a dataset directory.
    """
    experiments: Dict[str, ExperimentDirectory]
    scaler_exists: bool
    split_distribution_exists: bool
    hist_box_plot_exists: bool
    pie_plot_exists: bool
    categorical_stats_json_exists: bool
    continuous_stats_json_exists: bool
    correlation_matrix_exists: bool


@dataclasses.dataclass
class ExperimentGroupDirectory:
    """
    Represents the structure of an experiment group directory.
    """
    datasets: Dict[str, DatasetDirectory]


@dataclasses.dataclass
class ResultStructure:
    """
    Represents the structure of the results directory.
    """
    config_log: ConfigLog
    error_log: Optional[ErrorLog]
    html_report: Optional[HTMLReport]
    experiment_groups: Dict[str, ExperimentGroupDirectory]

    def __str__(self):
        report = []
        report.append("Result Structure:")
        report.append(f"Config Log Exists: {self.config_log.file_exists}")
        report.append(
            "Error Log Exists: "
            f"{self.error_log.file_exists if self.error_log else 'N/A'}" # pylint: disable=W1405
        )
        report.append(f"HTML Report Exists: {self.html_report.index_exists}")

        for group_name, group in self.experiment_groups.items():
            report.append(f"\nExperiment Group: {group_name}")
            for dataset_name, dataset in group.datasets.items():
                report.append(f"  \nDataset: {dataset_name}")
                report.append(f"    Scaler Exists: {dataset.scaler_exists}")
                report.append(
                    f"    Histogram Exists: {dataset.hist_box_plot_exists}"
                )
                report.append(f"    Pie Plot Exists: {dataset.pie_plot_exists}")
                report.append(
                    "    Categorical Stats JSON Exists: "
                    f"{dataset.categorical_stats_json_exists}"
                )
                report.append(
                    "    Continuous Stats JSON Exists: "
                    f"{dataset.continuous_stats_json_exists}"
                )
                report.append(
                    "    Correlation Matrix Exists: "
                    f"{dataset.correlation_matrix_exists}"
                )
                report.append(
                    "    Split Distribution Exists: "
                    f"{dataset.split_distribution_exists}"
                )

                for experiment_name, experiment in dataset.experiments.items():
                    report.append(f"    \nExperiment: {experiment_name}")
                    report.append(f"      Save Model: {experiment.save_model}")
                    report.append(
                        f"      Evaluate Model: {experiment.evaluate_model}"
                    )
                    report.append(
                        "      Evaluate Model CV: "
                        f"{experiment.evaluate_model_cv}"
                    )
                    report.append(
                        f"      Compare Models: {experiment.compare_models}"
                    )
                    report.append(
                        f"      Plot Pred vs Obs: {experiment.plot_pred_vs_obs}"
                    )
                    report.append(
                        "      Plot Learning Curve: "
                        f"{experiment.plot_learning_curve}"
                    )
                    report.append(
                        "      Plot Feature Importance: "
                        f"{experiment.plot_feature_importance}"
                    )
                    report.append(
                        f"      Plot Residuals: {experiment.plot_residuals}"
                    )
                    report.append(
                        "      Plot Model Comparison: "
                        f"{experiment.plot_model_comparison}"
                    )
                    report.append(
                        f"      Confusion Matrix: {experiment.confusion_matrix}"
                    )
                    report.append(
                        "      Plot Confusion Heatmap: "
                        f"{experiment.plot_confusion_heatmap}"
                    )
                    report.append(
                        f"      Plot ROC Curve: {experiment.plot_roc_curve}"
                    )
                    report.append(
                        "      Plot Precision Recall Curve: "
                        f"{experiment.plot_precision_recall_curve}"
                    )

        report.append("\nHTML Report:")
        report.append(f"  Index Exists: {self.html_report.index_exists}")
        report.append(
            f"  Index CSS Exists: {self.html_report.index_css_exists}"
        )
        report.append(
            f"  Experiment CSS Exists: {self.html_report.experiment_css_exists}"
        )
        report.append(
            f"  Dataset CSS Exists: {self.html_report.dataset_css_exists}"
        )

        report.append("  Dataset Pages:")
        for page, exists in self.html_report.dataset_pages.items():
            report.append(
                f"    {page}: {'Exists' if exists else 'Does not exist'}" # pylint: disable=W1405
            )

        report.append("  Experiment Pages:")
        for page, exists in self.html_report.experiment_pages.items():
            report.append(
                f"    {page}: {'Exists' if exists else 'Does not exist'}" # pylint: disable=W1405
            )

        return "\n".join(report)

    @classmethod
    def from_config(
        cls,
        config_manager: configuration_manager.ConfigurationManager,
        workflow_path: str
    ):
        config_log = ConfigLog(file_exists=True)
        error_log = ErrorLog(file_exists=True)
        workflow_methods = MethodCallVisitor.extract_workflow_methods(
            workflow_path
        )
        base_scale_method = (
            False if config_manager.base_data_manager.scale_method is None
            else True
        )
        # base_categorical_features = (
        #     False if not config_manager.base_data_manager.categorical_features
        #     else True
        # )
        experiment_groups = cls.get_experiment_groups(
            config_manager.experiment_groups, workflow_methods,
            base_scale_method, config_manager.categorical_features
        )
        dataset_pages, experiment_pages = cls.get_html_pages_from_groups(
            experiment_groups
        )
        html_report = HTMLReport(
            index_exists=True,
            index_css_exists=True,
            experiment_css_exists=True,
            dataset_css_exists=True,
            dataset_pages=dataset_pages,
            experiment_pages=experiment_pages
        )
        return cls(
            config_log=config_log,
            error_log=error_log,
            html_report=html_report,
            experiment_groups=experiment_groups
        )

    @classmethod
    def from_directory(cls, path: pathlib.Path):
        root_path = pathlib.Path(path)
        experiment_groups = {}
        dataset_pages = {}
        experiment_pages = {}
        config_log = ConfigLog(
            file_exists=root_path.joinpath("config_log.md").exists()
        )
        error_log = ErrorLog(
            file_exists=root_path.joinpath("error_log.txt").exists()
        )

        for group_dir in root_path.glob("*"):
            if not group_dir.is_dir() or group_dir.name == "html_report":
                continue

            group_name = group_dir.name
            datasets = {}

            for dataset_dir in group_dir.glob("*"):
                if not dataset_dir.is_dir():
                    continue

                dataset_name = dataset_dir.name
                dataset_file = f"{group_name}_{dataset_name}"
                dataset_pages[dataset_file] = root_path.joinpath(
                    f"html_report/{dataset_file}.html"
                ).exists()
                experiments = {}


                for exp_dir in dataset_dir.glob("*"):
                    if (
                        not exp_dir.is_dir() or
                        exp_dir.name == "split_distribution"
                    ):
                        continue

                    experiment_file = f"{dataset_name}_{exp_dir.name}"
                    experiment_pages[experiment_file] = root_path.joinpath(
                        f"html_report/{experiment_file}.html"
                    ).exists()
                    workflow_methods = cls.get_workflow_methods_from_dir(
                        exp_dir
                    )

                    experiments[exp_dir.name] = ExperimentDirectory(
                        "save_model" in workflow_methods,
                        "evaluate_model" in workflow_methods,
                        "evaluate_model_cv" in workflow_methods,
                        "compare_models" in workflow_methods,
                        "plot_pred_vs_obs" in workflow_methods,
                        "plot_learning_curve" in workflow_methods,
                        "plot_feature_importance" in workflow_methods,
                        "plot_residuals" in workflow_methods,
                        "plot_model_comparison" in workflow_methods,
                        "confusion_matrix" in workflow_methods,
                        "plot_confusion_heatmap" in workflow_methods,
                        "plot_roc_curve" in workflow_methods,
                        "plot_precision_recall_curve" in workflow_methods
                    )

                split_distribution_dir = dataset_dir / "split_distribution"
                hist_box_plot_dir = split_distribution_dir / "hist_box_plot"
                pie_plot_dir = split_distribution_dir / "pie_plot"

                hist_box_plot_exists = any(hist_box_plot_dir.glob("*.png"))
                pie_plot_exists = any(pie_plot_dir.glob("*.png"))
                categorical_stats_json_exists = split_distribution_dir.joinpath(
                    "categorical_stats.json"
                ).exists()
                continuous_stats_json_exists = split_distribution_dir.joinpath(
                    "continuous_stats.json"
                ).exists()
                correlation_matrix_exists = split_distribution_dir.joinpath(
                    "correlation_matrix.png"
                ).exists()
                scaler_exists = any(dataset_dir.glob("*.joblib"))

                datasets[dataset_dir.name] = DatasetDirectory(
                    experiments=experiments,
                    scaler_exists=scaler_exists,
                    split_distribution_exists=True,
                    hist_box_plot_exists=hist_box_plot_exists,
                    pie_plot_exists=pie_plot_exists,
                    categorical_stats_json_exists=categorical_stats_json_exists,
                    continuous_stats_json_exists=continuous_stats_json_exists,
                    correlation_matrix_exists=correlation_matrix_exists
                )

            experiment_groups[group_dir.name] = ExperimentGroupDirectory(
                datasets=datasets
            )

        html_report = HTMLReport(
            index_exists=root_path.joinpath(
                "html_report/index.html"
            ).exists(),
            index_css_exists=root_path.joinpath(
                "html_report/index.css"
            ).exists(),
            dataset_css_exists=root_path.joinpath(
                "html_report/dataset.css"
            ).exists(),
            experiment_css_exists=root_path.joinpath(
                "html_report/experiment.css"
            ).exists(),
            dataset_pages=dataset_pages,
            experiment_pages=experiment_pages,
        )

        return cls(
            config_log=config_log,
            error_log=error_log,
            html_report=html_report,
            experiment_groups=experiment_groups
        )

    @staticmethod
    def get_png_metadata(file_path: pathlib.Path) -> str:
        """Extract method name from PNG metadata."""
        with PIL.Image.open(file_path) as img:
            metadata = img.info
            return metadata.get("method", "")

    @staticmethod
    def get_json_metadata(file_path: pathlib.Path) -> str:
        """Extract method name from JSON metadata."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("_metadata", {}).get("method", "")

    @staticmethod
    def get_html_pages_from_groups(
        experiment_groups: Dict[str, ExperimentGroupDirectory]
    ) -> Tuple[Dict, Dict]:
        dataset_pages = {}
        experiment_pages = {}

        for group_name, group in experiment_groups.items():
            for dataset_name, dataset in group.datasets.items():
                dataset_page_name = f"{group_name}_{dataset_name}"
                dataset_pages[dataset_page_name] = True

                for exp_name in dataset.experiments.keys():
                    exp_page_name = f"{dataset_name}_{exp_name}"
                    experiment_pages[exp_page_name] = True

        return dataset_pages, experiment_pages

    @staticmethod
    def get_experiment_groups(
        groups: List[experiment_group.ExperimentGroup],
        workflow_methods: Set[str],
        base_scale_method: bool,
        categorical_features: Dict[str, List[str]]
    ):
        experiment_groups = {}

        for group in groups:
            datasets = {}
            for dataset_name in group.datasets:
                categorical_feat_names = categorical_features.get(
                    dataset_name, None
                )
                categorical_features_exist = bool(categorical_feat_names)

                if isinstance(dataset_name, tuple):
                    dataset_name = (
                        f"{dataset_name[0].split(".")[0]}_{dataset_name[1]}"
                    )
                experiments = {}
                for algorithm in group.algorithms:
                    if isinstance(algorithm, list):
                        algorithm = "_".join(algorithm)

                    experiment_name = f"{group.name}_{algorithm}"
                    experiments[experiment_name] = ExperimentDirectory(
                        "save_model" in workflow_methods,
                        "evaluate_model" in workflow_methods,
                        "evaluate_model_cv" in workflow_methods,
                        "compare_models" in workflow_methods,
                        "plot_pred_vs_obs" in workflow_methods,
                        "plot_learning_curve" in workflow_methods,
                        "plot_feature_importance" in workflow_methods,
                        "plot_residuals" in workflow_methods,
                        "plot_model_comparison" in workflow_methods,
                        "confusion_matrix" in workflow_methods,
                        "plot_confusion_heatmap" in workflow_methods,
                        "plot_roc_curve" in workflow_methods,
                        "plot_precision_recall_curve" in workflow_methods
                    )

                if group.data_config:
                    scaler = group.data_config.get(
                        "scale_method", base_scale_method
                    )
                    scaler_exists = True if scaler else False
                else:
                    scaler_exists = base_scale_method

                datasets[pathlib.Path(dataset_name).stem] = DatasetDirectory(
                    experiments=experiments.copy(),
                    scaler_exists=scaler_exists,
                    split_distribution_exists=True,
                    continuous_stats_json_exists=True,
                    hist_box_plot_exists=True,
                    categorical_stats_json_exists=categorical_features_exist,
                    pie_plot_exists=categorical_features_exist,
                    correlation_matrix_exists=True
                )

            experiment_groups[group.name] = ExperimentGroupDirectory(
                datasets=datasets.copy()
            )

        return experiment_groups

    @staticmethod
    def get_workflow_methods_from_dir(experiment_dir):
        workflow_methods = set()

        if experiment_dir.glob("*.pkl"):
            workflow_methods.add("save_model")

        for json_file in experiment_dir.glob("*.json"):
            method = ResultStructure.get_json_metadata(json_file)
            if method:
                workflow_methods.add(method)

        for png_file in experiment_dir.glob("*.png"):
            method = ResultStructure.get_png_metadata(png_file)
            if method:
                workflow_methods.add(method)

        return workflow_methods
