"""Collect services and data requried by all evaluators."""

import dataclasses
import logging

from brisk.evaluation.metric_manager import MetricManager
from brisk.evaluation.services.metadata import MetadataService
from brisk.evaluation.services.io import IOService
from brisk.evaluation.services.utility import UtilityService

@dataclasses.dataclass
class ServiceBundle:
    """Bundle of services that evaluators need access to."""
    metadata: MetadataService
    io: IOService
    utility: UtilityService
    metric_config: MetricManager
    logger: logging.Logger
