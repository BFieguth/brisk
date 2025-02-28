.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Description
   * - :class:`~brisk.training.alert_mailer.AlertMailer`
     - Sends notification emails using Gmail's SMTP service.
   * - :class:`~brisk.configuration.algorithm_wrapper.AlgorithmWrapper`
     - Wraps a machine learning algorithm and provides an interface using the algorithm.
   * - :class:`~brisk.configuration.configuration.Configuration`
     - Provide an interface for creating experiment groups.
   * - :class:`~brisk.configuration.configuration_manager.ConfigurationManager`
     - Process the ExperimentGroups and prepare the required DataManagers.
   * - :class:`~brisk.data.data_manager.DataManager`
     - Handles the grouping, splitting, and scaling of data. Arguments are used to define the splitting strategy.
   * - :class:`~brisk.data.data_split_info.DataSplitInfo`
     - Stores and analyzes training and testing datasets, providing methods for calculating descriptive statistics and visualizing feature distributions.
   * - :class:`~brisk.evaluation.evaluation_manager.EvaluationManager`
     - Provides methods for evaluating models and generating plots.
   * - :class:`~brisk.configuration.experiment.Experiment`
     - Stores all the data needed for one experiment run.
   * - :class:`~brisk.configuration.experiment_factory.ExperimentFactory`
     - Create a que of Experiments from an ExperimentGroup.
   * - :class:`~brisk.configuration.experiment_group.ExperimentGroup`
     - Groups experiments that will be run with the same settings.
   * - :class:`~brisk.training.logging_util.FileFormatter`
     - Formats log messages with a visual separator between log entries.
   * - :class:`~brisk.evaluation.metric_manager.MetricManager`
     - Stores MetricWrapper instances that define evaluation metrics.
   * - :class:`~brisk.evaluation.metric_wrapper.MetricWrapper`
     - Wraps a metric function and provides a convenient interface using the metric.
   * - :class:`~brisk.reporting.report_manager.ReportManager`
     - Generates HTML report from training results.
   * - :class:`~brisk.training.logging_util.TqdmLoggingHandler`
     - Logs messages to stdout or stderr using tqdm.
   * - :class:`~brisk.training.training_manager.TrainingManager`
     - Coordinates the training process, loading the data and running the experiments.
   * - :class:`~brisk.training.workflow.Workflow`
     - Defines the steps to take when training a model.