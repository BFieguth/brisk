"""Collect services and data requried by all evaluators."""

import dataclasses

from brisk.services.logging import LoggingService
from brisk.services.metadata import MetadataService
from brisk.services.io import IOService
from brisk.services.utility import UtilityService
from brisk.services.reporting import ReportingService

@dataclasses.dataclass
class ServiceBundle:
    """Bundle of services that evaluators need access to."""
    logger: LoggingService
    metadata: MetadataService
    io: IOService
    utility: UtilityService
    reporting: ReportingService
