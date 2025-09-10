.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Description
   * - :class:`~brisk.services.base.BaseService`
     - Abstract base class for all services in the Brisk framework.
   * - :class:`~brisk.services.rerun.CaptureStrategy`
     - Strategy for capturing experiment state for reruns.
   * - :class:`~brisk.services.rerun.CoordinatingStrategy`
     - Strategy for coordinating multiple rerun operations.
   * - :class:`~brisk.training.logging_util.FileFormatter`
     - Formats log messages with a visual separator between log entries.
   * - :class:`~brisk.training.logging_util.FileFormatter`
     - Formats log messages with a visual separator between log entries.
   * - :class:`~brisk.services.GlobalServiceManager`
     - Manages global service instances and dependencies.
   * - :class:`~brisk.services.io.IOService`
     - Provides file input/output operations and data serialization.
   * - :class:`~brisk.services.logging.LoggingService`
     - Manages logging configuration and handlers.
   * - :class:`~brisk.services.metadata.MetadataService`
     - Manages experiment metadata and versioning information.
   * - :class:`~brisk.services.io.NumpyEncoder`
     - JSON encoder for NumPy arrays and data types.
   * - :class:`~brisk.services.reporting.ReportingContext`
     - Context manager for report generation operations.
   * - :class:`~brisk.services.reporting.ReportingService`
     - Coordinates report generation and data collection.
   * - :class:`~brisk.services.rerun.RerunService`
     - Manages experiment rerun capabilities and strategies.
   * - :class:`~brisk.services.rerun.RerunStrategy`
     - Abstract base class for rerun strategies.
   * - :class:`~brisk.services.bundle.ServiceBundle`
     - Bundles related services together for easier management.
   * - :class:`~brisk.training.logging_util.TqdmLoggingHandler`
     - Logs messages to stdout or stderr using tqdm.
   * - :class:`~brisk.training.logging_util.TqdmLoggingHandler`
     - Logs messages to stdout or stderr using tqdm.
   * - :class:`~brisk.services.utility.UtilityService`
     - Provides common utility functions and helpers.