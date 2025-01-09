"""
This module contains the ResultStructure class, which is used to represent the
structure of the results directory of a Brisk experiment as booleans.
"""

import ast
import dataclasses
import json
import pathlib
from typing import Dict, Optional, Set, Tuple

import PIL

from brisk.configuration import configuration_manager

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
        base_categorical_features = (
            False if not config_manager.base_data_manager.categorical_features
            else True
        )
        experiment_groups = cls.get_experiment_groups(
            config_manager.experiment_groups, workflow_methods,
            base_scale_method, base_categorical_features
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
        groups,
        workflow_methods,
        base_scale_method,
        base_categorical_features
    ):
        experiment_groups = {}
        datasets = {}
        experiments = {}

        for group in groups:
            for dataset_name in group.datasets:
                for algorithm in group.algorithms:
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
                    categorical_features = group.data_config.get(
                        "categorical_features", base_categorical_features
                    )
                    categorical_features_exist = (
                        True if categorical_features else False
                    )
                else:
                    scaler_exists = base_scale_method
                    categorical_features_exist = base_categorical_features

                datasets[pathlib.Path(dataset_name).stem] = DatasetDirectory(
                    experiments=experiments,
                    scaler_exists=scaler_exists,
                    split_distribution_exists=True,
                    continuous_stats_json_exists=True,
                    hist_box_plot_exists=True,
                    categorical_stats_json_exists=categorical_features_exist,
                    pie_plot_exists=categorical_features_exist,
                    correlation_matrix_exists=True
                )

            experiment_groups[group.name] = ExperimentGroupDirectory(
                datasets=datasets
            )

        return experiment_groups

    @staticmethod
