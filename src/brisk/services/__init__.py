"""Global service management."""
from typing import Dict
from pathlib import Path

import numpy as np

from brisk.services import bundle, logging, metadata, io, utility
from brisk.configuration import algorithm_wrapper

class GlobalServiceManager:
    """Manages services that are available to the entire Brisk package."""
    def __init__(
        self,
        algorithm_config: algorithm_wrapper.AlgorithmCollection,
        results_dir: Path,
        verbose: bool = False
    ):
        self.services = {}
        self.services["logging"] = logging.LoggingService(
            "logging", None, verbose
        )
        self.services["metadata"] = metadata.MetadataService(
            "metadata", algorithm_config
        )
        self.services["io"] = io.IOService("io", results_dir, None)
        self.services["utility"] = utility.UtilityService(
            "utility", algorithm_config, None, None
        )
        self._register_services()

    def _register_services(self) -> None:
        """Register all services with each other so they can access each other."""
        for key, service in self.services.items():
            if hasattr(service, 'register_services'):
                other_services = self.services.copy()
                other_services.pop(key)
                service.register_services(other_services)

    def get_service_bundle(self) -> bundle.ServiceBundle:
        return bundle.ServiceBundle(
            logger=self.services['logging'],
            metadata=self.services['metadata'],
            io=self.services['io'],
            utility=self.services['utility']
        )

    def shutdown(self) -> None:
        """Implement cleanup here."""
        pass

    def update_utility_config(
        self,
        group_index_train=None,
        group_index_test=None,
    ) -> None:
        """Update utility service configuration."""
        if self.services['utility']:
            utility = self.services['utility']
            if group_index_train is not None and group_index_test is not None:
                utility.set_split_indices(group_index_train, group_index_test)


_global_service_manager = None

def initialize_services(
    algorithm_config: algorithm_wrapper.AlgorithmCollection,
    results_dir: Path,
    verbose: bool = False
) -> None:
    global _global_service_manager
    _global_service_manager = GlobalServiceManager(
        algorithm_config=algorithm_config,
        results_dir=results_dir,
        verbose=verbose
    )


def get_services() -> bundle.ServiceBundle:
    """Get global ServiceBundle instance"""
    global _global_service_manager

    if _global_service_manager is not None:
        return _global_service_manager.get_service_bundle()
    
    raise RuntimeError(
        "Services not initalized. Call initalize_services() first."
    )


def get_service_manager() -> GlobalServiceManager:
    """Get the global service manager."""
    global _global_service_manager
    if _global_service_manager is None:
        raise RuntimeError(
            "Services not initialized. Call initialize_services() first."
        )
    return _global_service_manager


def shutdown_services() -> None:
    """Shutdown and cleanup global services."""
    global _global_service_manager
    if _global_service_manager:
        _global_service_manager.shutdown()
        _global_service_manager = None


def is_initialized() -> bool:
    """Check if services are initialized."""
    global _global_service_manager
    return _global_service_manager is not None


def update_experiment_config(
    output_dir: str,
    group_index_train: Dict[str, np.array],
    group_index_test: Dict[str, np.array]
) -> None:
    """Update service configurations for a new experiment."""
    global _global_service_manager
    if _global_service_manager is None:
        raise RuntimeError("Services not initialized")
    
    _global_service_manager.services["io"].set_output_dir(
        Path(output_dir)
    )
    
    _global_service_manager.update_utility_config(
        group_index_train=group_index_train,
        group_index_test=group_index_test
    )
