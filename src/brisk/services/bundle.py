"""Collect services available to all classes at runtime."""

import dataclasses

from brisk.services import logging, metadata, io, utility, reporting

@dataclasses.dataclass
class ServiceBundle:
    """Bundle of services that classes can access at runtime.
    
    Parameters
    ----------
    logger (LoggingService): 
        The logging service.
    metadata (MetadataService): 
        The metadata service.
    io (IOService): 
        The I/O service.
    utility (UtilityService): 
        The utility service.
    reporting (ReportingService): 
        The reporting service.

    Returns
    -------
    None
    """
    logger: logging.LoggingService
    metadata: metadata.MetadataService
    io: io.IOService
    utility: utility.UtilityService
    reporting: reporting.ReportingService
