.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Description
   * - :class:`~brisk.training.logging_util.FileFormatter`
     - Formats log messages with a visual separator between log entries.
   * - :class:`~brisk.training.logging_util.TqdmLoggingHandler`
     - Logs messages to stdout or stderr using tqdm.
   * - :class:`~brisk.training.training_manager.TrainingManager`
     - Coordinates the training process, loading the data and running the experiments.
   * - :class:`~brisk.training.workflow.Workflow`
     - Defines the steps to take when training a model.