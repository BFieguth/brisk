"""Collect services and data requried by all evaluators."""

import dataclasses

from brisk.evaluation.services.logging import LoggingService
from brisk.evaluation.services.metadata import MetadataService
from brisk.evaluation.services.io import IOService
from brisk.evaluation.services.utility import UtilityService

@dataclasses.dataclass
class ServiceBundle:
    """Bundle of services that evaluators need access to."""
    logger: LoggingService
    metadata: MetadataService
    io: IOService
    utility: UtilityService
