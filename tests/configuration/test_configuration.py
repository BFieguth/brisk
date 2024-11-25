import pytest
from unittest.mock import Mock, patch
from brisk.configuration.Configuration import Configuration
from brisk.configuration.ConfigurationManager import ConfigurationManager

import pytest
from unittest.mock import patch
from brisk.configuration.Configuration import Configuration
from brisk.configuration.ConfigurationManager import ConfigurationManager

@pytest.fixture
def mock_validation():
    """Mock only the validation methods of ExperimentGroup"""
    with patch('brisk.configuration.ExperimentGroup.ExperimentGroup.__post_init__'):
        yield


@pytest.fixture
def basic_config(mock_project_root, monkeypatch):
    """Create a basic configuration with mocked project root."""
    monkeypatch.chdir(mock_project_root)
    (mock_project_root / 'algorithms.py').write_text("""
from sklearn.linear_model import LinearRegression, Ridge
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper

ALGORITHM_CONFIG = [
    AlgorithmWrapper(
        name="linear",
        display_name="Linear Regression",
        algorithm_class=LinearRegression
    ),
    AlgorithmWrapper(
        name="ridge",
        display_name="Ridge Regression",
        algorithm_class=Ridge
    )
]
""")

    return Configuration(default_algorithms=["linear", "ridge"])


class TestConfiguration:
    def test_initialization(self, basic_config):
        """Test configuration initialization"""
        assert basic_config.default_algorithms == ["linear", "ridge"]
        assert basic_config.experiment_groups == []

    def test_add_experiment_group(self, basic_config):
        """Test adding experiment group with defaults"""
        basic_config.add_experiment_group(
            name="test_group",
            datasets=["data.csv"]
        )
        
        group = basic_config.experiment_groups[0]
        assert group.name == "test_group"
        assert group.datasets == ["data.csv"]
        assert group.algorithms == ["linear", "ridge"]
        assert group.data_config is None
        assert group.algorithm_config is None

    def test_add_experiment_group_custom_algorithms(self, basic_config):
        """Test adding experiment group with custom algorithms"""
        algorithm_config = {"elasticnet": {"alpha": 0.5}}
        basic_config.add_experiment_group(
            name="custom_group",
            datasets=["data.csv"],
            algorithms=["elasticnet"],
            algorithm_config=algorithm_config
        )
        
        group = basic_config.experiment_groups[0]
        assert group.name == "custom_group"
        assert group.datasets == ["data.csv"]
        assert group.algorithms == ["elasticnet"]
        assert group.algorithm_config == algorithm_config

    def test_duplicate_name(self, basic_config):
        """Test adding experiment group with duplicate name"""
        # Add first group
        basic_config.add_experiment_group(
            name="test_group",
            datasets=["data.csv"]
        )
        
        # Attempt to add duplicate
        with pytest.raises(ValueError, match="already exists"):
            basic_config.add_experiment_group(
                name="test_group",
                datasets=["other.csv"]
            )

    def test_build_returns_configuration_manager(self, basic_config):
        """Test build method returns ConfigurationManager"""
        basic_config.add_experiment_group(
            name="test_group",
            datasets=["data.csv"]
        )
        
        manager = basic_config.build()
        assert isinstance(manager, ConfigurationManager)
