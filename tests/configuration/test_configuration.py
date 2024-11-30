import pytest
from brisk.configuration.configuration import Configuration
from brisk.configuration.configuration_manager import ConfigurationManager

@pytest.fixture
def configuration():
    """Create a basic configuration with mocked project root."""
    return Configuration(default_algorithms=["linear", "ridge"])


class TestConfiguration:
    def test_initialization(self, configuration):
        """Test configuration initialization"""
        assert configuration.default_algorithms == ["linear", "ridge"]
        assert configuration.experiment_groups == []

    def test_add_experiment_group(self, mock_regression_project, configuration):
        """Test adding experiment group with defaults"""
        configuration.add_experiment_group(
            name="test_group",
            datasets=["data.csv"]
        )
        
        group = configuration.experiment_groups[0]
        assert group.name == "test_group"
        assert group.datasets == ["data.csv"]
        assert group.algorithms == ["linear", "ridge"]
        assert group.data_config is None
        assert group.algorithm_config is None

    def test_add_experiment_group_custom_algorithms(self, mock_regression_project, configuration):
        """Test adding experiment group with custom algorithms"""
        algorithm_config = {"elasticnet": {"alpha": 0.5}}
        configuration.add_experiment_group(
            name="custom_group",
            datasets=["data.csv"],
            algorithms=["elasticnet"],
            algorithm_config=algorithm_config
        )
        
        group = configuration.experiment_groups[0]
        assert group.name == "custom_group"
        assert group.datasets == ["data.csv"]
        assert group.algorithms == ["elasticnet"]
        assert group.algorithm_config == algorithm_config

    def test_duplicate_name(self, mock_regression_project, configuration):
        """Test adding experiment group with duplicate name"""
        # Add first group
        configuration.add_experiment_group(
            name="test_group",
            datasets=["data.csv"]
        )
        
        # Attempt to add duplicate
        with pytest.raises(ValueError, match="already exists"):
            configuration.add_experiment_group(
                name="test_group",
                datasets=["other.csv"]
            )

    def test_build_returns_configuration_manager(self, mock_regression_project, configuration):
        """Test build method returns ConfigurationManager"""
        configuration.add_experiment_group(
            name="test_group",
            datasets=["data.csv"]
        )
        
        manager = configuration.build()
        assert isinstance(manager, ConfigurationManager)
