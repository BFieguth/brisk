"""
Utility methods required by evaluators are provided as instances of BaseService.
"""

from abc import ABC
from pathlib import Path

class BaseService(ABC):
    """Base class for services in EvaluationManager."""
    def __init__(self, output_dir: Path, logger=None):
        self.output_dir = output_dir
        self.logger = logger
