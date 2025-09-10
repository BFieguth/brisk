"""Base service class for the Brisk services architecture.

This module provides the BaseService abstract base class that serves as the
foundation for all services in the Brisk package. It implements common
functionality for service registration and cross-service communication.

The BaseService class provides a standardized interface for services to
register with each other and access other services by name, enabling loose
coupling between different service components.

Classes
-------
BaseService
    Abstract base class for all services in the Brisk package

Examples
--------
>>> from brisk.services.base import BaseService
>>> 
>>> class MyService(BaseService):
...     def __init__(self, name: str):
...         super().__init__(name)
...     
...     def do_something(self):
...         # Access another service
...         other_service = self.get_service("logging")
...         other_service.info("Hello from MyService")
>>> 
>>> # Create and register services
>>> my_service = MyService("my_service")
>>> other_services = {"logging": logging_service}
>>> my_service.register_services(other_services)
"""

from abc import ABC
from typing import Dict, Any, Optional

class BaseService(ABC):
    """Abstract base class for all services in the Brisk package.
    
    This class provides the foundation for all services in the Brisk
    architecture. It implements common functionality for service registration
    and cross-service communication, allowing services to access each other by
    name without tight coupling.
    
    The BaseService class enables the service registry pattern, where services
    can register references to other services and access them dynamically at
    runtime. This promotes loose coupling and makes the service architecture
    more flexible and maintainable.
    
    Attributes
    ----------
    name : str
        The unique name identifier for this service
    _other_services : Dict[str, Any]
        Dictionary mapping service names to service instances that this
        service can access
        
    Notes
    -----
    This is an abstract base class and should not be instantiated directly.
    Concrete service classes should inherit from BaseService and implement
    their specific functionality.
    
    Examples
    --------
    >>> from brisk.services.base import BaseService
    >>> 
    >>> class LoggingService(BaseService):
    ...     def __init__(self, name: str):
    ...         super().__init__(name)
    ...     
    ...     def log_info(self, message: str):
    ...         print(f"[{self.name}] {message}")
    >>> 
    >>> class DataService(BaseService):
    ...     def __init__(self, name: str):
    ...         super().__init__(name)
    ...     
    ...     def process_data(self):
    ...         # Access logging service
    ...         logger = self.get_service("logging")
    ...         logger.log_info("Processing data...")
    """
    def __init__(self, name: str) -> None:
        """Initialize the BaseService with a name.
        
        This constructor sets up the basic service infrastructure including
        the service name and an empty registry for other services.
        
        Parameters
        ----------
        name : str
            The unique name identifier for this service
            
        Examples
        --------
        >>> class MyService(BaseService):
        ...     def __init__(self, name: str):
        ...         super().__init__(name)
        >>> 
        >>> service = MyService("my_service")
        >>> print(service.name)  # "my_service"
        """
        self.name = name
        self._other_services: Dict[str, Any] = {}

    def register_services(self, services: Dict[str, Any]) -> None:
        """Register other services this service can access.
        
        This method allows the service to register references to other services,
        enabling cross-service communication. The registered services can then
        be accessed using the `get_service` method.
        
        Parameters
        ----------
        services : Dict[str, Any]
            Dictionary mapping service names to service instances that this
            service can access
            
        Notes
        -----
        This method replaces any previously registered services. If you need
        to add services incrementally, you should merge the new services with
        the existing `_other_services` dictionary.
        
        Examples
        --------
        >>> service = MyService("my_service")
        >>> other_services = {
        ...     "logging": logging_service,
        ...     "metadata": metadata_service
        ... }
        >>> service.register_services(other_services)
        >>> 
        >>> # Now the service can access other services
        >>> logger = service.get_service("logging")
        """
        self._other_services = services

    def get_service(self, service_name: str) -> Optional[Any]:
        """Get another service by name.
        
        This method retrieves a registered service by its name. It provides
        a safe way for services to access other services without tight coupling.
        
        Parameters
        ----------
        service_name : str
            The name of the service to retrieve
            
        Returns
        -------
        Optional[Any]
            The service instance if found, None otherwise
            
        Raises
        ------
        KeyError
            If the requested service is not registered
            
        Examples
        --------
        >>> service = MyService("my_service")
        >>> service.register_services({"logging": logging_service})
        >>> 
        >>> # Get a registered service
        >>> logger = service.get_service("logging")
        >>> logger.info("Hello from my service")
        >>> 
        >>> # Try to get a non-existent service
        >>> try:
        ...     other = service.get_service("nonexistent")
        ... except KeyError as e:
        ...     print(f"Service not found: {e}")
        """
        if service_name not in self._other_services:
            raise KeyError(
                f"Service {service_name} not found. "
                f"Registered services are: {list(self._other_services.keys())}"
            )
        return self._other_services.get(service_name)
