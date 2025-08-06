"""
Utility methods required by evaluators are provided as instances of BaseService.
"""

from abc import ABC
from typing import Dict, Any, Optional

class BaseService(ABC):
    """Base class for services in EvaluationManager."""
    def __init__(self, name: str):
        self.name = name
        self._other_services: Dict[str, Any] = {}

    def register_services(self, services: Dict[str, Any]) -> None:
        """Register other services this service can access."""
        self._other_services = services

    def get_service(self, service_name: str) -> Optional[Any]:
        """Get another service by name."""
        return self._other_services.get(service_name)
