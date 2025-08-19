from pathlib import Path

import pytest
import numpy as np

from brisk.services import (
    GlobalServiceManager, get_services, get_service_manager, is_initialized
)
from brisk.services.bundle import ServiceBundle
from conftest import get_algorithm_config, get_metric_config

@pytest.fixture
def algo_config():
    return get_algorithm_config()


@pytest.fixture
def metric_config():
    return get_metric_config()


class TestGlobalServiceManager:
    def test_init(self, algo_config, metric_config, tmp_path):
        manager = GlobalServiceManager(
            algo_config, metric_config, Path(tmp_path / "results"), False
        )
        assert len(manager.services) == 5
        assert manager.is_initalized is True

    def test_singleton(self, algo_config, metric_config, tmp_path):
        manager = GlobalServiceManager(
            algo_config, metric_config, Path(tmp_path / "results"), False
        )
        manager.__setattr__("test_attribute", "check this value")
        assert manager.test_attribute == "check this value"

        # Should return the same instance with "test_attribute"
        manager2 = GlobalServiceManager(
            algo_config, metric_config, Path(tmp_path / "results"), False
        )
        assert manager2.test_attribute == "check this value"
        assert manager == manager2
        
        # Check test_attribute is gone after reset
        manager.reset()
        manager3 = GlobalServiceManager(
            algo_config, metric_config, Path(tmp_path / "results"), False
        )
        with pytest.raises(
            AttributeError,
            match="'GlobalServiceManager' object has no attribute 'test_attribute'"
        ):
            attribute = manager3.test_attribute

    def test_update_utility_config(self, algo_config, metric_config, tmp_path):
        manager = GlobalServiceManager(
            algo_config, metric_config, Path(tmp_path / "results"), False
        )
        group_index_train = {"feature": np.array([1, 2, 3, 4, 5])}
        group_index_test = {"feature": np.array([6, 7, 8])}
        manager.update_utility_config(group_index_train, group_index_test)

        utility_service = manager.services["utility"]
        assert utility_service.group_index_train == group_index_train
        assert utility_service.group_index_test == group_index_test

        manager.reset()

        manager = GlobalServiceManager(
            algo_config, metric_config, Path(tmp_path / "results"), False
        )
        utility_service = manager.services["utility"]
        assert utility_service.group_index_train is None
        assert utility_service.group_index_test is None

    def test_get_services(self, algo_config, metric_config, tmp_path):
        with pytest.raises(RuntimeError, match="Services not initialized"):
            services = get_services()

        manager = GlobalServiceManager(
            algo_config, metric_config, Path(tmp_path / "results"), False
        )
        services = get_services()
        assert isinstance(services, ServiceBundle)

    def test_get_service_manager(self, algo_config, metric_config, tmp_path):
        with pytest.raises(RuntimeError, match="Services not initialized."):
            service_manager = get_service_manager()

        manager = GlobalServiceManager(
            algo_config, metric_config, Path(tmp_path / "results"), False
        )
        service_manager = get_service_manager()
        assert manager == service_manager

    def test_is_initialized(self, algo_config, metric_config, tmp_path):
        assert is_initialized() is False
        manager = GlobalServiceManager(
            algo_config, metric_config, Path(tmp_path / "results"), False
        )
        assert is_initialized() is True
