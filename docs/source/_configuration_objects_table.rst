.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Description
   * - :class:`~brisk.configuration.algorithm_wrapper.AlgorithmWrapper`
     - Wraps a machine learning algorithm and provides an interface using the algorithm.
   * - :class:`~brisk.configuration.configuration.Configuration`
     - Provide an interface for creating experiment groups.
   * - :class:`~brisk.configuration.configuration_manager.ConfigurationManager`
     - Process the ExperimentGroups and prepare the required DataManagers.
   * - :class:`~brisk.configuration.experiment.Experiment`
     - Stores all the data needed for one experiment run.
   * - :class:`~brisk.configuration.experiment_factory.ExperimentFactory`
     - Create a que of Experiments from an ExperimentGroup.
   * - :class:`~brisk.configuration.experiment_group.ExperimentGroup`
     - Groups experiments that will be run with the same settings.
   * - :class:`~brisk.configuration.project.find_project_root`
     - Finds the project root directory containing .briskconfig.