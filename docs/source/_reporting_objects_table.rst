.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Description
   * - :class:`~brisk.data.data_manager.DataManager`
     - Handles data splitting and preprocessing pipelines. Arguments are used to define the splitting strategy and preprocessing steps.
   * - :class:`~brisk.reporting.report_data.Dataset`
     - Represents a dataset in the report.
   * - :class:`~brisk.configuration.experiment.Experiment`
     - Stores all the data needed for one experiment run.
   * - :class:`~brisk.configuration.experiment_group.ExperimentGroup`
     - Groups experiments that will be run with the same settings.
   * - :class:`~brisk.reporting.report_data.FeatureDistribution`
     - Data structure for feature distribution information.
   * - :class:`~brisk.reporting.report_data.Navbar`
     - Navigation bar configuration for reports.
   * - :class:`~brisk.reporting.report_data.PlotData`
     - Represents plot data and metadata for reports.
   * - :class:`~brisk.reporting.report_data.ReportData`
     - Container for all report data and metadata.
   * - :class:`~brisk.reporting.report_renderer.ReportRenderer`
     - Renders HTML reports from training results.
   * - :class:`~brisk.reporting.report_data.RoundedModel`
     - Base model with automatic number rounding for display.
   * - :class:`~brisk.reporting.report_data.TableData`
     - Represents tabular data for report display.