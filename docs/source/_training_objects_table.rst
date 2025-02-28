.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Description
   * - :class:`~brisk.training.alert_mailer.AlertMailer`
     - Sends notification emails using Gmail's SMTP service.
   * - :class:`~brisk.training.training_manager.TrainingManager`
     - Coordinates the training process, loading the data and running the experiments.
   * - :class:`~brisk.training.workflow.Workflow`
     - Defines the steps to take when training a model.