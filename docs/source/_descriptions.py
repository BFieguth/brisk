"""Descriptions of Brisk objects for documentation."""

from collections import OrderedDict

# Maintain descriptions in alphabetical order by object name
DESCRIPTIONS = OrderedDict({
    "AlgorithmCollection": {
        "path": "~brisk.configuration.algorithm_collection.AlgorithmCollection",
        "desc": "A collection of AlgorithmWrappers."
    },
    "AlgorithmWrapper": {
        "path": "~brisk.configuration.algorithm_wrapper.AlgorithmWrapper",
        "desc": "Wraps a machine learning algorithm and provides an interface using the algorithm."
    },
    "BarPlot": {
        "path": "~brisk.evaluation.evaluators.builtin.dataset_plots.BarPlot",
        "desc": "Creates bar plots for categorical features."
    },
    "BaseEvaluator": {
        "path": "~brisk.evaluation.evaluators.base.BaseEvaluator",
        "desc": "Abstract base class for all evaluators."
    },
    "BaseService": {
        "path": "~brisk.services.base.BaseService",
        "desc": "Abstract base class for all services in the Brisk framework."
    },
    "CategoricalStatistics": {
        "path": "~brisk.evaluation.evaluators.builtin.dataset_measures.CategoricalStatistics",
        "desc": "Computes statistics for categorical variables in datasets."
    },
    "CaptureStrategy": {
        "path": "~brisk.services.rerun.CaptureStrategy",
        "desc": "Strategy for capturing experiment state for reruns."
    },
    "cli_helpers": {
        "path": "~brisk.cli.cli_helpers",
        "desc": "Provides helper functions for the CLI."
    },
    "check_env": {
        "path": "~brisk.cli.cli.check_env",
        "desc": "Checks the environment compatibility with a previous run."
    },
    "create": {
        "path": "~brisk.cli.cli.create",
        "desc": "Creates a new project."
    },
    "create_data": {
        "path": "~brisk.cli.cli.create_data",
        "desc": "Creates a synthetic dataset."
    },
    "CompareModels": {
        "path": "~brisk.evaluation.evaluators.builtin.common_measures.CompareModels",
        "desc": "Compares performance across multiple models."
    },
    "CoordinatingStrategy": {
        "path": "~brisk.services.rerun.CoordinatingStrategy",
        "desc": "Strategy for coordinating multiple rerun operations."
    },
    "Configuration": {
        "path": "~brisk.configuration.configuration.Configuration",
        "desc": "Provide an interface for creating experiment groups."
    },
    "ConfigurationManager": {
        "path": "~brisk.configuration.configuration_manager.ConfigurationManager",
        "desc": "Process the ExperimentGroups and prepare the required DataManagers."
    },
    "ConfusionMatrix": {
        "path": "~brisk.evaluation.evaluators.builtin.classification_measures.ConfusionMatrix",
        "desc": "Computes confusion matrix for classification models."
    },
    "ContinuousStatistics": {
        "path": "~brisk.evaluation.evaluators.builtin.dataset_measures.ContinuousStatistics",
        "desc": "Computes statistics for continuous variables in datasets."
    },
    "CorrelationMatrix": {
        "path": "~brisk.evaluation.evaluators.builtin.dataset_plots.CorrelationMatrix",
        "desc": "Creates correlation matrix plots."
    },
    "DataManager": {
        "path": "~brisk.data.data_manager.DataManager",
        "desc": "Handles data splitting and preprocessing pipelines. Arguments"
                " are used to define the splitting strategy and preprocessing steps."
    },
    "DataSplits": {
        "path": "~brisk.data.data_splits.DataSplits",
        "desc": "Stores DataSplitInfo instances."
    },
    "DataSplitInfo": {
        "path": "~brisk.data.data_split_info.DataSplitInfo",
        "desc": "Stores and analyzes training and testing datasets, providing methods for calculating "
                "descriptive statistics and visualizing feature distributions."
    },
    "Dataset": {
        "path": "~brisk.reporting.report_data.Dataset",
        "desc": "Represents a dataset in the report."
    },
    "DatasetMeasureEvaluator": {
        "path": "~brisk.evaluation.evaluators.dataset_measure_evaluator.DatasetMeasureEvaluator",
        "desc": "Base class for evaluators that compute dataset-level measures."
    },
    "DatasetPlotEvaluator": {
        "path": "~brisk.evaluation.evaluators.dataset_plot_evaluator.DatasetPlotEvaluator",
        "desc": "Base class for evaluators that create dataset-level plots."
    },
    "EnvironmentDiff": {
        "path": "~brisk.cli.environment.EnvironmentDiff",
        "desc": "Represents the differences between environments."
    },
    "EnvironmentManager": {
        "path": "~brisk.cli.environment.EnvironmentManager",
        "desc": "Manages environment capture, comparison, and export for reproducible runs."
    },
    "EvaluateModel": {
        "path": "~brisk.evaluation.evaluators.builtin.common_measures.EvaluateModel",
        "desc": "Evaluates model performance using specified metrics."
    },
    "EvaluateModelCV": {
        "path": "~brisk.evaluation.evaluators.builtin.common_measures.EvaluateModelCV",
        "desc": "Evaluates model performance using cross-validation."
    },
    "EvaluatorRegistry": {
        "path": "~brisk.evaluation.evaluators.registry.EvaluatorRegistry",
        "desc": "Registry for managing and discovering evaluators."
    },
    "EvaluationManager": {
        "path": "~brisk.evaluation.evaluation_manager.EvaluationManager",
        "desc": "Provides methods for evaluating models and generating plots."
    },
    "Experiment": {
        "path": "~brisk.configuration.experiment.Experiment",
        "desc": "Stores all the data needed for one experiment run."
    },
    "ExperimentFactory": {
        "path": "~brisk.configuration.experiment_factory.ExperimentFactory",
        "desc": "Create a que of Experiments from an ExperimentGroup."
    },
    "ExperimentGroup": {
        "path": "~brisk.configuration.experiment_group.ExperimentGroup",
        "desc": "Groups experiments that will be run with the same settings."
    },
    "export_env": {
        "path": "~brisk.cli.cli.export_env",
        "desc": "Create a requirements.txt file from the environment captured during a previous experiment run."
    },
    "FeatureDistribution": {
        "path": "~brisk.reporting.report_data.FeatureDistribution",
        "desc": "Data structure for feature distribution information."
    },
    "FileFormatter": {
        "path": "~brisk.training.logging_util.FileFormatter",
        "desc": "Formats log messages with a visual separator between log entries."
    },
    "find_project_root": {
        "path": "~brisk.configuration.project.find_project_root",
        "desc": "Finds the project root directory containing .briskconfig."
    },
    "GlobalServiceManager": {
        "path": "~brisk.services.GlobalServiceManager",
        "desc": "Manages global service instances and dependencies."
    },
    "Navbar": {
        "path": "~brisk.reporting.report_data.Navbar",
        "desc": "Navigation bar configuration for reports."
    },
    "NumpyEncoder": {
        "path": "~brisk.services.io.NumpyEncoder",
        "desc": "JSON encoder for NumPy arrays and data types."
    },
    "PackageInfo": {
        "path": "~brisk.cli.environment.PackageInfo",
        "desc": "Information about a package and its version."
    },
    "PickleJSONDecoder": {
        "path": "~brisk.theme.theme_serializer.PickleJSONDecoder",
        "desc": "JSON decoder that handles pickled objects."
    },
    "PickleJSONEncoder": {
        "path": "~brisk.theme.theme_serializer.PickleJSONEncoder",
        "desc": "JSON encoder that handles pickled objects."
    },
    "PlotData": {
        "path": "~brisk.reporting.report_data.PlotData",
        "desc": "Represents plot data and metadata for reports."
    },
    "PlotSettings": {
        "path": "~brisk.theme.plot_settings.PlotSettings",
        "desc": "Configuration for plot appearance and styling."
    },
    "Histogram": {
        "path": "~brisk.evaluation.evaluators.builtin.dataset_plots.Histogram",
        "desc": "Creates histogram plots for dataset features."
    },
    "HyperparameterTuning": {
        "path": "~brisk.evaluation.evaluators.builtin.optimization.HyperparameterTuning",
        "desc": "Performs hyperparameter optimization."
    },
    "IOService": {
        "path": "~brisk.services.io.IOService",
        "desc": "Provides file input/output operations and data serialization."
    },
    "load_data": {
        "path": "~brisk.cli.cli.load_data",
        "desc": "Load a scikit-learn dataset by name."
    },
    "load_sklearn_dataset": {
        "path": "~brisk.cli.cli_helpers.load_sklearn_dataset",
        "desc": "Load a scikit-learn dataset by name."
    },
    "LoggingService": {
        "path": "~brisk.services.logging.LoggingService",
        "desc": "Manages logging configuration and handlers."
    },
    "MeasureEvaluator": {
        "path": "~brisk.evaluation.evaluators.measure_evaluator.MeasureEvaluator",
        "desc": "Base class for evaluators that compute model performance measures."
    },
    "MetricManager": {
        "path": "~brisk.evaluation.metric_manager.MetricManager",
        "desc": "Stores MetricWrapper instances that define evaluation metrics."
    },
    "MetricWrapper": {
        "path": "~brisk.evaluation.metric_wrapper.MetricWrapper",
        "desc": "Wraps a metric function and provides a convenient interface using the metric."
    },
    "MetadataService": {
        "path": "~brisk.services.metadata.MetadataService",
        "desc": "Manages experiment metadata and versioning information."
    },
    "PlotConfusionHeatmap": {
        "path": "~brisk.evaluation.evaluators.builtin.classification_plots.PlotConfusionHeatmap",
        "desc": "Creates confusion matrix heatmap plots."
    },
    "PlotEvaluator": {
        "path": "~brisk.evaluation.evaluators.plot_evaluator.PlotEvaluator",
        "desc": "Base class for evaluators that create model performance plots."
    },
    "PlotFeatureImportance": {
        "path": "~brisk.evaluation.evaluators.builtin.common_plots.PlotFeatureImportance",
        "desc": "Creates feature importance plots."
    },
    "PlotLearningCurve": {
        "path": "~brisk.evaluation.evaluators.builtin.common_plots.PlotLearningCurve",
        "desc": "Creates learning curve plots."
    },
    "PlotModelComparison": {
        "path": "~brisk.evaluation.evaluators.builtin.common_plots.PlotModelComparison",
        "desc": "Creates model comparison plots."
    },
    "PlotPredVsObs": {
        "path": "~brisk.evaluation.evaluators.builtin.regression_plots.PlotPredVsObs",
        "desc": "Creates predicted vs observed plots for regression models."
    },
    "PlotPrecisionRecallCurve": {
        "path": "~brisk.evaluation.evaluators.builtin.classification_plots.PlotPrecisionRecallCurve",
        "desc": "Creates precision-recall curve plots."
    },
    "PlotResiduals": {
        "path": "~brisk.evaluation.evaluators.builtin.regression_plots.PlotResiduals",
        "desc": "Creates residual plots for regression models."
    },
    "PlotRocCurve": {
        "path": "~brisk.evaluation.evaluators.builtin.classification_plots.PlotRocCurve",
        "desc": "Creates ROC curve plots for classification models."
    },
    "PlotShapleyValues": {
        "path": "~brisk.evaluation.evaluators.builtin.common_plots.PlotShapleyValues",
        "desc": "Creates SHAP value plots for model interpretability."
    },
    "ReportData": {
        "path": "~brisk.reporting.report_data.ReportData",
        "desc": "Container for all report data and metadata."
    },
    "ReportingContext": {
        "path": "~brisk.services.reporting.ReportingContext",
        "desc": "Context manager for report generation operations."
    },
    "ReportingService": {
        "path": "~brisk.services.reporting.ReportingService",
        "desc": "Coordinates report generation and data collection."
    },
    "ReportRenderer": {
        "path": "~brisk.reporting.report_renderer.ReportRenderer",
        "desc": "Renders HTML reports from training results."
    },
    "RerunService": {
        "path": "~brisk.services.rerun.RerunService",
        "desc": "Manages experiment rerun capabilities and strategies."
    },
    "RerunStrategy": {
        "path": "~brisk.services.rerun.RerunStrategy",
        "desc": "Abstract base class for rerun strategies."
    },
    "RoundedModel": {
        "path": "~brisk.reporting.report_data.RoundedModel",
        "desc": "Base model with automatic number rounding for display."
    },
    "run": {
        "path": "~brisk.cli.cli.run",
        "desc": "Run the current experiment setup."
    },
    "ServiceBundle": {
        "path": "~brisk.services.bundle.ServiceBundle",
        "desc": "Bundles related services together for easier management."
    },
    "TableData": {
        "path": "~brisk.reporting.report_data.TableData",
        "desc": "Represents tabular data for report display."
    },
    "ThemePickleJSONSerializer": {
        "path": "~brisk.theme.theme_serializer.ThemePickleJSONSerializer",
        "desc": "Serializes theme objects using pickle and JSON encoding."
    },
    "TqdmLoggingHandler": {
        "path": "~brisk.training.logging_util.TqdmLoggingHandler",
        "desc": "Logs messages to stdout or stderr using tqdm."
    },
    "TrainingManager": {
        "path": "~brisk.training.training_manager.TrainingManager",
        "desc": "Coordinates the training process, loading the data and running the experiments."
    },
    "UtilityService": {
        "path": "~brisk.services.utility.UtilityService",
        "desc": "Provides common utility functions and helpers."
    },
    "VersionMatch": {
        "path": "~brisk.cli.environment.VersionMatch",
        "desc": "Enumeration of version matching states."
    },
    "Workflow": {
        "path": "~brisk.training.workflow.Workflow",
        "desc": "Defines the steps to take when training a model."
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
