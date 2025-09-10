.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Description
   * - :class:`~brisk.cli.environment.EnvironmentDiff`
     - Represents the differences between environments.
   * - :class:`~brisk.cli.environment.EnvironmentManager`
     - Manages environment capture, comparison, and export for reproducible runs.
   * - :class:`~brisk.cli.environment.VersionMatch`
     - Enumeration of version matching states.
   * - :class:`~brisk.cli.cli.check_env`
     - Checks the environment compatibility with a previous run.
   * - :class:`~brisk.cli.cli.create`
     - Creates a new project.
   * - :class:`~brisk.cli.cli.create_data`
     - Creates a synthetic dataset.
   * - :class:`~brisk.cli.cli.export_env`
     - Create a requirements.txt file from the environment captured during a previous experiment run.
   * - :class:`~brisk.cli.cli.load_data`
     - Load a scikit-learn dataset by name.
   * - :class:`~brisk.cli.cli_helpers.load_sklearn_dataset`
     - Load a scikit-learn dataset by name.
   * - :class:`~brisk.cli.cli.run`
     - Run the current experiment setup.