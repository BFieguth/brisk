"""Descriptions of Brisk objects for documentation."""

from collections import OrderedDict

# Maintain descriptions in alphabetical order by object name
DESCRIPTIONS = OrderedDict({
    "DataManager": {
        "path": "~brisk.data.data_manager.DataManager",
        "desc": "Handles data splitting and preprocessing pipelines. Arguments"
                " are used to define the splitting strategy and preprocessing steps."
    },
    "DataSplitInfo": {
        "path": "~brisk.data.data_split_info.DataSplitInfo",
        "desc": "Stores and analyzes training and testing datasets, providing methods for calculating "
                "descriptive statistics and visualizing feature distributions."
    },
    "Configuration": {
        "path": "~brisk.configuration.configuration.Configuration",
        "desc": "Provide an interface for creating experiment groups."
    },
    "ConfigurationManager": {
        "path": "~brisk.configuration.configuration_manager.ConfigurationManager",
        "desc": "Process the ExperimentGroups and prepare the required DataManagers."
    },
    "ExperimentFactory": {
        "path": "~brisk.configuration.experiment_factory.ExperimentFactory",
        "desc": "Create a que of Experiments from an ExperimentGroup."
    },
    "ExperimentGroup": {
        "path": "~brisk.configuration.experiment_group.ExperimentGroup",
        "desc": "Groups experiments that will be run with the same settings."
    },
    "Experiment": {
        "path": "~brisk.configuration.experiment.Experiment",
        "desc": "Stores all the data needed for one experiment run."
    },
    "EvaluationManager": {
        "path": "~brisk.evaluation.evaluation_manager.EvaluationManager",
        "desc": "Provides methods for evaluating models and generating plots."
    },
    "MetricManager": {
        "path": "~brisk.evaluation.metric_manager.MetricManager",
        "desc": "Stores MetricWrapper instances that define evaluation metrics."
    },
    "ReportManager": {
        "path": "~brisk.reporting.report_manager.ReportManager",
        "desc": "Generates HTML report from training results."
    },
    "TrainingManager": {
        "path": "~brisk.training.training_manager.TrainingManager",
        "desc": "Coordinates the training process, loading the data and running the experiments."
    },
    "Workflow": {
        "path": "~brisk.training.workflow.Workflow",
        "desc": "Defines the steps to take when training a model."
    },
    "MetricWrapper": {
        "path": "~brisk.evaluation.metric_wrapper.MetricWrapper",
        "desc": "Wraps a metric function and provides a convenient interface using the metric."
    },
    "AlgorithmWrapper": {
        "path": "~brisk.configuration.algorithm_wrapper.AlgorithmWrapper",
        "desc": "Wraps a machine learning algorithm and provides an interface using the algorithm."
    },
    "TqdmLoggingHandler": {
        "path": "~brisk.training.logging_util.TqdmLoggingHandler",
        "desc": "Logs messages to stdout or stderr using tqdm."
    },
    "FileFormatter": {
        "path": "~brisk.training.logging_util.FileFormatter",
        "desc": "Formats log messages with a visual separator between log entries."
    },
    "find_project_root": {
        "path": "~brisk.configuration.project.find_project_root",
        "desc": "Finds the project root directory containing .briskconfig."
    },
})

def generate_list_table(objects=None):
    """Generate RST list-table for specified objects or all objects."""
    if objects is None:
        objects = DESCRIPTIONS.keys()
    
    rows = []
    for name in sorted(objects):
        obj = DESCRIPTIONS[name]
        rows.append(f"   * - :class:`{obj['path']}`\n     - {obj['desc']}")
    
    return "\n".join([
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 30 70",
        "",
        "   * - Object",
        "     - Description",
    ] + rows)
