"""Global service management for the Brisk package.

This module provides a centralized service management system that makes various
services available throughout the Brisk package. It implements a singleton
pattern to ensure consistent access to logging, metadata, I/O, utility,
reporting, and rerun services across the entire application.

The GlobalServiceManager acts as a central hub that initializes and manages all
services, providing a unified interface for accessing functionality like
logging, file I/O, metadata storage, and report generation.

Classes
-------
GlobalServiceManager
    Singleton class that manages all services and provides access to them

Functions
---------
initialize_services
    Initialize the global service manager with configuration
get_services
    Get the global ServiceBundle instance
get_service_manager
    Get the global service manager instance
is_initialized
    Check if services are initialized
update_experiment_config
    Update service configurations for a new experiment

Examples
--------
>>> from brisk.services import initialize_services, get_services
>>> from pathlib import Path
>>> 
>>> # Initialize services
>>> results_dir = Path("results")
>>> initialize_services(results_dir, verbose=True)
>>> 
>>> # Get service bundle
>>> services = get_services()
>>> services.logger.info("Services initialized")
>>> 
>>> # Update configuration for new experiment
>>> update_experiment_config(
...     output_dir="experiment_1",
...     group_index_train={"train": np.array([0, 1, 2])},
...     group_index_test={"test": np.array([3, 4, 5])}
... )
"""
from typing import Dict, Optional, Any
import pathlib

import numpy as np

from brisk.services import (
    bundle, logging, metadata, io, utility, reporting, rerun
)


class GlobalServiceManager:
    """A singleton that makes services available to the entire Brisk package.
    
    This class implements the singleton pattern to ensure that only one instance
    of the service manager exists throughout the application lifecycle. It
    manages all core services including logging, metadata, I/O, utility,
    reporting, and rerun functionality.
    
    The service manager provides a centralized way for all components of the
    Brisk package to access shared functionality without tight coupling.
    Services can register with each other to enable cross-service communication.
    
    Attributes
    ----------
    services : Dict[str, Service]
        Dictionary mapping service names to service instances
    instance : Optional[GlobalServiceManager]
        Class variable holding the singleton instance
    is_initalized : bool
        Flag indicating whether the service manager has been initialized
        
    Notes
    -----
    This class uses the singleton pattern, so multiple instantiations will
    return the same instance. The class should be initialized once at the
    start of the application using `initialize_services()`.
    
    Examples
    --------
    >>> from brisk.services import initialize_services, get_service_manager
    >>> from pathlib import Path
    >>> 
    >>> # Initialize the service manager
    >>> initialize_services(Path("results"), verbose=True)
    >>> 
    >>> # Get the singleton instance
    >>> manager = get_service_manager()
    >>> print(manager.is_initalized)  # True
    >>> 
    >>> # Access services
    >>> services = manager.get_service_bundle()
    >>> services.logger.info("Hello from service manager")
    """
    instance: Optional["GlobalServiceManager"] = None
    is_initalized: bool = False

    def __new__(cls, *args, **kwargs) -> "GlobalServiceManager":
        """Create or return the singleton instance.
        
        This method implements the singleton pattern by ensuring only one
        instance of GlobalServiceManager exists throughout the application.
        
        Returns
        -------
        GlobalServiceManager
            The singleton instance of the service manager
        """
        if cls.instance is None:
            cls.instance = super(GlobalServiceManager, cls).__new__(cls)
        return cls.instance

    def __init__(
        self,
        results_dir: pathlib.Path,
        verbose: bool = False,
        mode: str = "capture",
        rerun_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the GlobalServiceManager and all services.
        
        This constructor initializes all core services and sets up the service
        registry. It only runs once per singleton instance to prevent
        re-initialization.
        
        Parameters
        ----------
        results_dir : pathlib.Path
            The root directory for storing results and logs
        verbose : bool, default=False
            Whether to enable verbose logging output
        mode : str, default="capture"
            The mode for the RerunService ("capture" or "coordinate")
        rerun_config : Optional[Dict[str, Any]], default=None
            Configuration dictionary for the RerunService
            
        Notes
        -----
        The constructor will only initialize services once, even if called
        multiple times. This prevents re-initialization of the singleton.
        """
        if self.is_initalized:
            return

        self.services = {}
        self.services["logging"] = logging.LoggingService(
            "logging", results_dir, verbose
        )
        self.services["metadata"] = metadata.MetadataService("metadata")
        self.services["io"] = io.IOService("io", results_dir, None)
        self.services["utility"] = utility.UtilityService(
            "utility", None, None
        )
        self.services["reporting"] = reporting.ReportingService(
            "reporting"
        )
        self.services["rerun"] = rerun.RerunService("rerun", mode, rerun_config)
        self._register_services()
        self.is_initalized = True

    def _register_services(self) -> None:
        """Register all services so they can access each other.
        
        This method enables cross-service communication by allowing each service
        to register references to other services. This is useful for services
        that need to interact with each other during operation.
        
        Notes
        -----
        Only services that implement a `register_services` method will be
        registered with other services. This prevents errors for services that
        don't need cross-service communication.
        """
        for key, service in self.services.items():
            if hasattr(service, "register_services"):
                other_services = self.services.copy()
                other_services.pop(key)
                service.register_services(other_services)

    def get_service_bundle(self) -> bundle.ServiceBundle:
        """Get the service bundle containing all services.
        
        This method creates and returns a ServiceBundle object that provides
        convenient access to all managed services through a single interface.
        
        Returns
        -------
        bundle.ServiceBundle
            A bundle containing all service instances
            
        Examples
        --------
        >>> manager = get_service_manager()
        >>> services = manager.get_service_bundle()
        >>> services.logger.info("Using service bundle")
        >>> services.metadata.store("key", "value")
        """
        return bundle.ServiceBundle(
            logger=self.services["logging"],
            metadata=self.services["metadata"],
            io=self.services["io"],
            utility=self.services["utility"],
            reporting=self.services["reporting"],
            rerun=self.services["rerun"],
        )

    def update_utility_config(
        self,
        group_index_train: Optional[Dict[str, np.ndarray]] = None,
        group_index_test: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Update utility service configuration with group indices.
        
        This method updates the utility service with training and test group
        indices, which are used for data splitting and group-based operations.
        
        Parameters
        ----------
        group_index_train : Optional[Dict[str, np.ndarray]], default=None
            Dictionary mapping group names to training indices
        group_index_test : Optional[Dict[str, np.ndarray]], default=None
            Dictionary mapping group names to test indices
            
        Notes
        -----
        Both parameters must be provided for the configuration to be updated.
        If either is None, no update will be performed.
        
        Examples
        --------
        >>> manager = get_service_manager()
        >>> manager.update_utility_config(
        ...     group_index_train={"train": np.array([0, 1, 2])},
        ...     group_index_test={"test": np.array([3, 4, 5])}
        ... )
        """
        if self.services["utility"]:
            utility_service = self.services["utility"]
            if group_index_train is not None and group_index_test is not None:
                utility_service.set_split_indices(
                    group_index_train, group_index_test
                )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for testing or cleanup.
        
        This method resets the singleton instance and initialization flag,
        allowing for clean re-initialization. This is primarily useful for
        testing scenarios where you need to reset the service state.
        
        Notes
        -----
        This method should be used with caution in production code as it
        will destroy the current service state and require re-initialization.
        
        Examples
        --------
        >>> # Reset for testing
        >>> GlobalServiceManager.reset()
        >>> # Now services are uninitialized
        >>> print(is_initialized())  # False
        """
        cls.instance = None
        cls.is_initalized = False


def initialize_services(
    results_dir: pathlib.Path,
    verbose: bool = False,
    mode: str = "capture",
    rerun_config: Optional[Dict[str, Any]] = None
) -> None:
    """Initialize the global service manager.

    This function creates and initializes the GlobalServiceManager singleton
    with the specified configuration. It should be called once at the start
    of the application to set up all services.

    Parameters
    ----------
    results_dir : pathlib.Path
        The root directory for storing results, logs, and other outputs
    verbose : bool, default=False
        Whether to enable verbose logging output
    mode : str, default="capture"
        The mode for the RerunService ("capture" or "coordinate")
    rerun_config : Optional[Dict[str, Any]], default=None
        Configuration dictionary for the RerunService

    Notes
    -----
    This function should be called before using any other service functions.
    Calling it multiple times will not re-initialize the services due to
    the singleton pattern.

    Examples
    --------
    >>> from brisk.services import initialize_services
    >>> from pathlib import Path
    >>> 
    >>> # Initialize with default settings
    >>> initialize_services(Path("results"))
    >>> 
    >>> # Initialize with verbose logging
    >>> initialize_services(Path("results"), verbose=True)
    >>> 
    >>> # Initialize with custom rerun configuration
    >>> rerun_config = {"max_retries": 3, "timeout": 300}
    >>> initialize_services(
    ...     Path("results"), 
    ...     mode="coordinate", 
    ...     rerun_config=rerun_config
    ... )
    """
    GlobalServiceManager(
        results_dir=results_dir,
        verbose=verbose,
        mode=mode,
        rerun_config=rerun_config
    )


def get_services() -> bundle.ServiceBundle:
    """Get the global ServiceBundle instance.

    This function returns a ServiceBundle containing all initialized services,
    providing convenient access to logging, metadata, I/O, utility, reporting,
    and rerun functionality.

    Returns
    -------
    bundle.ServiceBundle
        A bundle containing all service instances

    Raises
    ------
    RuntimeError
        If services have not been initialized yet

    Examples
    --------
    >>> from brisk.services import initialize_services, get_services
    >>> from pathlib import Path
    >>> 
    >>> # Initialize services first
    >>> initialize_services(Path("results"))
    >>> 
    >>> # Get service bundle
    >>> services = get_services()
    >>> services.logger.info("Hello from services")
    >>> services.metadata.store("key", "value")
    """
    if GlobalServiceManager.instance is not None:
        return GlobalServiceManager.instance.get_service_bundle()

    raise RuntimeError(
        "Services not initialized. Call initialize_services() first."
    )


def get_service_manager() -> GlobalServiceManager:
    """Get the global service manager instance.

    This function returns the singleton GlobalServiceManager instance,
    which provides direct access to the service management functionality.

    Returns
    -------
    GlobalServiceManager
        The global service manager singleton instance

    Raises
    ------
    RuntimeError
        If services have not been initialized yet

    Examples
    --------
    >>> from brisk.services import initialize_services, get_service_manager
    >>> from pathlib import Path
    >>> 
    >>> # Initialize services first
    >>> initialize_services(Path("results"))
    >>> 
    >>> # Get service manager
    >>> manager = get_service_manager()
    >>> print(manager.is_initalized)  # True
    >>> services = manager.get_service_bundle()
    """
    if GlobalServiceManager.instance is None:
        raise RuntimeError(
            "Services not initialized. Call initialize_services() first."
        )
    return GlobalServiceManager.instance


def is_initialized() -> bool:
    """Check if services are initialized.

    This function checks whether the global service manager has been
    initialized and is ready for use.

    Returns
    -------
    bool
        True if services are initialized, False otherwise

    Examples
    --------
    >>> from brisk.services import is_initialized, initialize_services
    >>> from pathlib import Path
    >>> 
    >>> print(is_initialized())  # False
    >>> initialize_services(Path("results"))
    >>> print(is_initialized())  # True
    """
    return GlobalServiceManager.instance is not None


def update_experiment_config(
    output_dir: str,
    group_index_train: Dict[str, np.ndarray],
    group_index_test: Dict[str, np.ndarray]
) -> None:
    """Update service configurations for a new experiment.

    This function updates the I/O service output directory and utility service
    group indices for a new experiment. It should be called when starting
    a new experiment to ensure services are configured correctly.

    Parameters
    ----------
    output_dir : str
        The output directory path for the experiment
    group_index_train : Dict[str, np.ndarray]
        Dictionary mapping group names to training indices
    group_index_test : Dict[str, np.ndarray]
        Dictionary mapping group names to test indices

    Raises
    ------
    RuntimeError
        If services have not been initialized yet

    Examples
    --------
    >>> from brisk.services import initialize_services, update_experiment_config
    >>> from pathlib import Path
    >>> import numpy as np
    >>> 
    >>> # Initialize services first
    >>> initialize_services(Path("results"))
    >>> 
    >>> # Update configuration for new experiment
    >>> update_experiment_config(
    ...     output_dir="experiment_1",
    ...     group_index_train={"train": np.array([0, 1, 2, 3])},
    ...     group_index_test={"test": np.array([4, 5, 6, 7])}
    ... )
    """
    if GlobalServiceManager.instance is None:
        raise RuntimeError(
            "Services not initialized. Call initialize_services() first."
        )

    GlobalServiceManager.instance.services["io"].set_output_dir(
        pathlib.Path(output_dir)
    )

    GlobalServiceManager.instance.update_utility_config(
        group_index_train=group_index_train,
        group_index_test=group_index_test
    )
