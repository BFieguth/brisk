"""Service bundle for collecting and accessing all services at runtime.

This module provides the ServiceBundle dataclass that serves as a convenient
container for all services in the Brisk package. It allows classes to access
all services through a single object without needing to manage individual
service references.

The ServiceBundle acts as a facade pattern, providing a simplified interface
to the complex service architecture while maintaining loose coupling between
components.

Examples
--------
>>> from brisk.services import get_services
>>> 
>>> # Get the service bundle
>>> services = get_services()
>>> 
>>> # Use services through the bundle
>>> services.logger.info("Hello from service bundle")
>>> services.metadata.store("key", "value")
>>> services.io.save_plot(Path("plot.png"), plot=my_plot)
"""

import dataclasses

from brisk.services import logging, metadata, io, utility, reporting, rerun

@dataclasses.dataclass
class ServiceBundle:
    """Bundle of services that classes can access at runtime.
    
    This dataclass serves as a container for all services in the Brisk package,
    providing convenient access to logging, metadata, I/O, utility, reporting,
    and rerun functionality through a single object.
    
    The ServiceBundle implements the facade pattern, simplifying access to the
    service architecture while maintaining loose coupling between components.
    It is typically created by the GlobalServiceManager and accessed through
    the `get_services()` function.
    
    Attributes
    ----------
    logger : logging.LoggingService
        Service for logging messages and debugging information
    metadata : metadata.MetadataService
        Service for storing and retrieving metadata
    io : io.IOService
        Service for file I/O operations, data loading, and plot saving
    utility : utility.UtilityService
        Service for utility functions and data processing
    reporting : reporting.ReportingService
        Service for generating reports and storing report data
    rerun : rerun.RerunService
        Service for managing experiment reruns and coordination
        
    Notes
    -----
    This is a dataclass, so all attributes are automatically generated
    as instance variables. The dataclass provides a clean interface for
    accessing all services without tight coupling.
    
    Examples
    --------
    >>> from brisk.services import get_services
    >>> from pathlib import Path
    >>> 
    >>> # Get service bundle
    >>> services = get_services()
    >>> 
    >>> # Use different services
    >>> services.logger.info("Starting experiment")
    >>> services.metadata.store("experiment_id", "exp_001")
    >>> 
    >>> # Save data and plots
    >>> data = {"accuracy": 0.95, "precision": 0.92}
    >>> services.io.save_to_json(data, Path("results.json"), {})
    >>> 
    >>> # Generate report
    >>> services.reporting.generate_report()
    """
    logger: logging.LoggingService
    metadata: metadata.MetadataService
    io: io.IOService
    utility: utility.UtilityService
    reporting: reporting.ReportingService
    rerun: rerun.RerunService
