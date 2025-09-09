.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Description
   * - :class:`~brisk.configuration.algorithm_collection.AlgorithmCollection`
     - A collection of AlgorithmWrappers.
   * - :class:`~brisk.configuration.algorithm_wrapper.AlgorithmWrapper`
     - Wraps a machine learning algorithm and provides an interface using the algorithm.
   * - :class:`~brisk.evaluation.evaluators.builtin.dataset_plots.BarPlot`
     - Creates bar plots for categorical features.
   * - :class:`~brisk.evaluation.evaluators.base.BaseEvaluator`
     - Abstract base class for all evaluators.
   * - :class:`~brisk.services.base.BaseService`
     - Abstract base class for all services in the Brisk framework.
   * - :class:`~brisk.services.rerun.CaptureStrategy`
     - Strategy for capturing experiment state for reruns.
   * - :class:`~brisk.evaluation.evaluators.builtin.dataset_measures.CategoricalStatistics`
     - Computes statistics for categorical variables in datasets.
   * - :class:`~brisk.evaluation.evaluators.builtin.common_measures.CompareModels`
     - Compares performance across multiple models.
   * - :class:`~brisk.configuration.configuration.Configuration`
     - Provide an interface for creating experiment groups.
   * - :class:`~brisk.configuration.configuration_manager.ConfigurationManager`
     - Process the ExperimentGroups and prepare the required DataManagers.
   * - :class:`~brisk.evaluation.evaluators.builtin.classification_measures.ConfusionMatrix`
     - Computes confusion matrix for classification models.
   * - :class:`~brisk.evaluation.evaluators.builtin.dataset_measures.ContinuousStatistics`
     - Computes statistics for continuous variables in datasets.
   * - :class:`~brisk.services.rerun.CoordinatingStrategy`
     - Strategy for coordinating multiple rerun operations.
   * - :class:`~brisk.evaluation.evaluators.builtin.dataset_plots.CorrelationMatrix`
     - Creates correlation matrix plots.
   * - :class:`~brisk.data.data_manager.DataManager`
     - Handles data splitting and preprocessing pipelines. Arguments are used to define the splitting strategy and preprocessing steps.
   * - :class:`~brisk.data.data_split_info.DataSplitInfo`
     - Stores and analyzes training and testing datasets, providing methods for calculating descriptive statistics and visualizing feature distributions.
   * - :class:`~brisk.data.data_splits.DataSplits`
     - Stores DataSplitInfo instances.
   * - :class:`~brisk.reporting.report_data.Dataset`
     - Represents a dataset in the report.
   * - :class:`~brisk.evaluation.evaluators.dataset_measure_evaluator.DatasetMeasureEvaluator`
     - Base class for evaluators that compute dataset-level measures.
   * - :class:`~brisk.evaluation.evaluators.dataset_plot_evaluator.DatasetPlotEvaluator`
     - Base class for evaluators that create dataset-level plots.
   * - :class:`~brisk.cli.environment.EnvironmentDiff`
     - Represents the differences between environments.
   * - :class:`~brisk.cli.environment.EnvironmentManager`
     - Manages environment capture, comparison, and export for reproducible runs.
   * - :class:`~brisk.evaluation.evaluators.builtin.common_measures.EvaluateModel`
     - Evaluates model performance using specified metrics.
   * - :class:`~brisk.evaluation.evaluators.builtin.common_measures.EvaluateModelCV`
     - Evaluates model performance using cross-validation.
   * - :class:`~brisk.evaluation.evaluation_manager.EvaluationManager`
     - Provides methods for evaluating models and generating plots.
   * - :class:`~brisk.evaluation.evaluators.registry.EvaluatorRegistry`
     - Registry for managing and discovering evaluators.
   * - :class:`~brisk.configuration.experiment.Experiment`
     - Stores all the data needed for one experiment run.
   * - :class:`~brisk.configuration.experiment_factory.ExperimentFactory`
     - Create a que of Experiments from an ExperimentGroup.
   * - :class:`~brisk.configuration.experiment_group.ExperimentGroup`
     - Groups experiments that will be run with the same settings.
   * - :class:`~brisk.reporting.report_data.FeatureDistribution`
     - Data structure for feature distribution information.
   * - :class:`~brisk.training.logging_util.FileFormatter`
     - Formats log messages with a visual separator between log entries.
   * - :class:`~brisk.services.GlobalServiceManager`
     - Manages global service instances and dependencies.
   * - :class:`~brisk.evaluation.evaluators.builtin.dataset_plots.Histogram`
     - Creates histogram plots for dataset features.
   * - :class:`~brisk.evaluation.evaluators.builtin.optimization.HyperparameterTuning`
     - Performs hyperparameter optimization.
   * - :class:`~brisk.services.io.IOService`
     - Provides file input/output operations and data serialization.
   * - :class:`~brisk.services.logging.LoggingService`
     - Manages logging configuration and handlers.
   * - :class:`~brisk.evaluation.evaluators.measure_evaluator.MeasureEvaluator`
     - Base class for evaluators that compute model performance measures.
   * - :class:`~brisk.services.metadata.MetadataService`
     - Manages experiment metadata and versioning information.
   * - :class:`~brisk.evaluation.metric_manager.MetricManager`
     - Stores MetricWrapper instances that define evaluation metrics.
   * - :class:`~brisk.evaluation.metric_wrapper.MetricWrapper`
     - Wraps a metric function and provides a convenient interface using the metric.
   * - :class:`~brisk.reporting.report_data.Navbar`
     - Navigation bar configuration for reports.
   * - :class:`~brisk.services.io.NumpyEncoder`
     - JSON encoder for NumPy arrays and data types.
   * - :class:`~brisk.cli.environment.PackageInfo`
     - Information about a package and its version.
   * - :class:`~brisk.theme.theme_serializer.PickleJSONDecoder`
     - JSON decoder that handles pickled objects.
   * - :class:`~brisk.theme.theme_serializer.PickleJSONEncoder`
     - JSON encoder that handles pickled objects.
   * - :class:`~brisk.evaluation.evaluators.builtin.classification_plots.PlotConfusionHeatmap`
     - Creates confusion matrix heatmap plots.
   * - :class:`~brisk.reporting.report_data.PlotData`
     - Represents plot data and metadata for reports.
   * - :class:`~brisk.evaluation.evaluators.plot_evaluator.PlotEvaluator`
     - Base class for evaluators that create model performance plots.
   * - :class:`~brisk.evaluation.evaluators.builtin.common_plots.PlotFeatureImportance`
     - Creates feature importance plots.
   * - :class:`~brisk.evaluation.evaluators.builtin.common_plots.PlotLearningCurve`
     - Creates learning curve plots.
   * - :class:`~brisk.evaluation.evaluators.builtin.common_plots.PlotModelComparison`
     - Creates model comparison plots.
   * - :class:`~brisk.evaluation.evaluators.builtin.classification_plots.PlotPrecisionRecallCurve`
     - Creates precision-recall curve plots.
   * - :class:`~brisk.evaluation.evaluators.builtin.regression_plots.PlotPredVsObs`
     - Creates predicted vs observed plots for regression models.
   * - :class:`~brisk.evaluation.evaluators.builtin.regression_plots.PlotResiduals`
     - Creates residual plots for regression models.
   * - :class:`~brisk.evaluation.evaluators.builtin.classification_plots.PlotRocCurve`
     - Creates ROC curve plots for classification models.
   * - :class:`~brisk.theme.plot_settings.PlotSettings`
     - Configuration for plot appearance and styling.
   * - :class:`~brisk.evaluation.evaluators.builtin.common_plots.PlotShapleyValues`
     - Creates SHAP value plots for model interpretability.
   * - :class:`~brisk.reporting.report_data.ReportData`
     - Container for all report data and metadata.
   * - :class:`~brisk.reporting.report_renderer.ReportRenderer`
     - Renders HTML reports from training results.
   * - :class:`~brisk.services.reporting.ReportingContext`
     - Context manager for report generation operations.
   * - :class:`~brisk.services.reporting.ReportingService`
     - Coordinates report generation and data collection.
   * - :class:`~brisk.services.rerun.RerunService`
     - Manages experiment rerun capabilities and strategies.
   * - :class:`~brisk.services.rerun.RerunStrategy`
     - Abstract base class for rerun strategies.
   * - :class:`~brisk.reporting.report_data.RoundedModel`
     - Base model with automatic number rounding for display.
   * - :class:`~brisk.services.bundle.ServiceBundle`
     - Bundles related services together for easier management.
   * - :class:`~brisk.reporting.report_data.TableData`
     - Represents tabular data for report display.
   * - :class:`~brisk.theme.theme_serializer.ThemePickleJSONSerializer`
     - Serializes theme objects using pickle and JSON encoding.
   * - :class:`~brisk.training.logging_util.TqdmLoggingHandler`
     - Logs messages to stdout or stderr using tqdm.
   * - :class:`~brisk.training.training_manager.TrainingManager`
     - Coordinates the training process, loading the data and running the experiments.
   * - :class:`~brisk.services.utility.UtilityService`
     - Provides common utility functions and helpers.
   * - :class:`~brisk.cli.environment.VersionMatch`
     - Enumeration of version matching states.
   * - :class:`~brisk.training.workflow.Workflow`
     - Defines the steps to take when training a model.
   * - :class:`~brisk.cli.cli.check_env`
     - Checks the environment compatibility with a previous run.
   * - :class:`~brisk.cli.cli_helpers`
     - Provides helper functions for the CLI.
   * - :class:`~brisk.cli.cli.create`
     - Creates a new project.
   * - :class:`~brisk.cli.cli.create_data`
     - Creates a synthetic dataset.
   * - :class:`~brisk.cli.cli.export_env`
     - Create a requirements.txt file from the environment captured during a previous experiment run.
   * - :class:`~brisk.configuration.project.find_project_root`
     - Finds the project root directory containing .briskconfig.
   * - :class:`~brisk.cli.cli.load_data`
     - Load a scikit-learn dataset by name.
   * - :class:`~brisk.cli.cli_helpers.load_sklearn_dataset`
     - Load a scikit-learn dataset by name.
   * - :class:`~brisk.cli.cli.run`
     - Run the current experiment setup.