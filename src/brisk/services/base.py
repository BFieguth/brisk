"""
Utility methods required by evaluators are provided as instances of BaseService.
"""

from abc import ABC
from typing import Dict, Any, Optional

class BaseService(ABC):
    """Base class for services in EvaluationManager.
    
    Parameters
    ----------
    name (str): 
        The name of the service.

    Attributes
    ----------
    name (str): 
        The name of the service.
    _other_services (Dict[str, Any]): 
        A dictionary of other services this service can access.
    """
    def __init__(self, name: str):
        self.name = name
        self._other_services: Dict[str, Any] = {}

    def register_services(self, services: Dict[str, Any]) -> None:
        """Register other services this service can access.
        
        Parameters
        ----------
        services (Dict[str, Any]): 
            A dictionary of other services this service can access.

        Returns
        -------
        None
        """
        self._other_services = services

    def get_service(self, service_name: str) -> Optional[Any]:
        """Get another service by name.
        
        Parameters
        ----------
        service_name (str): 
            The name of the service to get.

        Returns
        -------
        Any: 
            The service instance.
        """
        if service_name not in self._other_services:
            raise KeyError(
                f"Service {service_name} not found. "
                f"Registered services are: {self._other_services.keys()}"
            )
        return self._other_services.get(service_name)
