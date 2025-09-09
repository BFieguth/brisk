.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Description
   * - :class:`~brisk.evaluation.evaluators.base.BaseEvaluator`
     - Abstract base class for all evaluators.
   * - :class:`~brisk.evaluation.evaluators.dataset_measure_evaluator.DatasetMeasureEvaluator`
     - Base class for evaluators that compute dataset-level measures.
   * - :class:`~brisk.evaluation.evaluators.dataset_plot_evaluator.DatasetPlotEvaluator`
     - Base class for evaluators that create dataset-level plots.
   * - :class:`~brisk.evaluation.evaluation_manager.EvaluationManager`
     - Provides methods for evaluating models and generating plots.
   * - :class:`~brisk.evaluation.evaluators.registry.EvaluatorRegistry`
     - Registry for managing and discovering evaluators.
   * - :class:`~brisk.evaluation.evaluators.measure_evaluator.MeasureEvaluator`
     - Base class for evaluators that compute model performance measures.
   * - :class:`~brisk.evaluation.metric_manager.MetricManager`
     - Stores MetricWrapper instances that define evaluation metrics.
   * - :class:`~brisk.evaluation.metric_wrapper.MetricWrapper`
     - Wraps a metric function and provides a convenient interface using the metric.
   * - :class:`~brisk.evaluation.evaluators.plot_evaluator.PlotEvaluator`
     - Base class for evaluators that create model performance plots.