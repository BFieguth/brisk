"""
This module contains the ResultStructure class, which is used to represent the
structure of the results directory of a Brisk experiment as booleans.
"""

import ast
import dataclasses
from typing import Dict, Optional, Set, Tuple

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
