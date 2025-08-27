import pytest

from brisk.configuration.configuration import Configuration
from brisk.configuration.experiment_group import ExperimentGroup
from brisk.configuration.configuration_manager import ConfigurationManager

@pytest.fixture
def configuration():
    """Create a basic configuration with mocked project root."""
    return Configuration(
        default_algorithms=["linear", "ridge"],
        default_workflow="regression_workflow"
    )


@pytest.fixture
def configuration_with_categorical_features():
    """Create a configuration with categorical features."""
    return Configuration(
        default_algorithms=["linear", "ridge"],
        default_workflow="regression_workflow",
        categorical_features={"categorical": ["category"]}
    )


@pytest.fixture
def configuration_with_workflow_args():
    """Create a configuration with workflow args."""
    return Configuration(
        default_algorithms=["linear", "ridge"],
        default_workflow="regression_workflow",
        default_workflow_args={"kfold": 5}
    )


@pytest.fixture
def configuration_algorithm_groups():
    """Create a configuration with algorithm groups."""
    return Configuration(
        default_algorithms=[["linear", "ridge"], ["linear", "elasticnet"]],
        default_workflow="regression_workflow"
    )


class TestConfiguration:
    def test_initialization(
        self,
        configuration,
        configuration_with_categorical_features,
        configuration_with_workflow_args,
        configuration_algorithm_groups
    ):
        """Test configuration initialization"""
        # configuration
        assert configuration.default_algorithms == ["linear", "ridge"]
        assert configuration.experiment_groups == []
        assert configuration.default_workflow_args == {}

        # configuration with categorical features
        assert configuration_with_categorical_features.default_algorithms == ["linear", "ridge"]
        assert configuration_with_categorical_features.categorical_features == {"categorical": ["category"]}
        assert configuration_with_categorical_features.experiment_groups == []
        assert configuration_with_categorical_features.default_workflow_args == {}

        # configuration with workflow args
        assert configuration_with_workflow_args.default_algorithms == ["linear", "ridge"]
        assert configuration_with_workflow_args.categorical_features == {}
        assert configuration_with_workflow_args.experiment_groups == []
        assert configuration_with_workflow_args.default_workflow_args == {"kfold": 5}

        # configuration with algorithm groups
        assert configuration_algorithm_groups.default_algorithms == [["linear", "ridge"], ["linear", "elasticnet"]]
        assert configuration_algorithm_groups.categorical_features == {}
        assert configuration_algorithm_groups.experiment_groups == []
        assert configuration_algorithm_groups.default_workflow_args == {}

    def test_add_experiment_group(self, mock_brisk_project, configuration):
        """Test adding experiment group with defaults"""
        configuration.add_experiment_group(
            name="test_group",
            datasets=["regression.csv"]
        )
        
        group = configuration.experiment_groups[0]
        assert group.name == "test_group"
        assert group.datasets == ["regression.csv"]
        assert group.algorithms == ["linear", "ridge"]
        assert group.data_config == {}
        assert group.algorithm_config is None
        assert group.description == ""
        assert group.workflow_args == {}

    def test_add_experiment_group_custom_algorithms(
        self,
        mock_brisk_project,
        configuration
    ):
        """Test adding experiment group with custom algorithms"""
        algorithm_config = {"elasticnet": {"alpha": 0.5}}
        configuration.add_experiment_group(
            name="custom_group",
            datasets=["regression.csv"],
            algorithms=["elasticnet"],
            algorithm_config=algorithm_config,
            description="This is a test description"
        )
        
        group = configuration.experiment_groups[0]
        assert group.name == "custom_group"
        assert group.datasets == ["regression.csv"]
        assert group.algorithms == ["elasticnet"]
        assert group.algorithm_config == algorithm_config
        assert group.description == "This is a test description"
        assert group.workflow_args == {}

    def test_duplicate_name(self, mock_brisk_project, configuration):
        """Test adding experiment group with duplicate name"""
        # Add first group
        configuration.add_experiment_group(
            name="test_group",
            datasets=["regression.csv"]
        )
        
        # Attempt to add duplicate
        with pytest.raises(ValueError, match="already exists"):
            configuration.add_experiment_group(
                name="test_group",
                datasets=["regression.csv"]
            )

    def test_add_experiment_workflow_args_missing_key(
        self,
        mock_brisk_project,
        configuration_with_workflow_args
    ):
        """Test adding experiment group with workflow args"""
        with pytest.raises(ValueError, match="workflow_args must have the same keys as defined in default_workflow_args"):
            configuration_with_workflow_args.add_experiment_group(
                name="test_group",
                datasets=["regression.csv"],
                workflow_args={"kfold": 10, "metrics": ["MAE", "R2"]}
            )

    def test_add_experiment_workflow_args(
        self,
        mock_brisk_project,
        configuration_with_workflow_args
    ):
        """Test adding experiment group with workflow args"""
        configuration_with_workflow_args.add_experiment_group(
            name="test_group",
            datasets=["regression.csv"],
            workflow_args={"kfold": 10}
        )
        
        group = configuration_with_workflow_args.experiment_groups[0]
        assert group.name == "test_group"
        assert group.datasets == ["regression.csv"]
        assert group.algorithms == ["linear", "ridge"]
        assert group.data_config == {}
        assert group.algorithm_config is None
        assert group.description == ""
        assert group.workflow_args == {"kfold": 10}

    def test_build_returns_configuration_manager(
        self,
        mock_brisk_project,
        configuration
    ):
        """Test build method returns ConfigurationManager"""
        configuration.add_experiment_group(
            name="test_group",
            datasets=["regression.csv"]
        )
        
        manager = configuration.build()
        assert isinstance(manager, ConfigurationManager)
        assert manager.experiment_groups == configuration.experiment_groups
        assert manager.categorical_features == configuration.categorical_features

    def test_check_name_exists(self, mock_brisk_project, configuration):
        """Test check_name_exists method"""
        configuration.experiment_groups = [
            ExperimentGroup(name="group", workflow="regression_workflow", datasets=["regression.csv"]),
            ExperimentGroup(name="group_2", workflow="regression_workflow", datasets=["regression.csv"]),
            ExperimentGroup(name="group_3", workflow="regression_workflow", datasets=["regression.csv"])
        ]
        with pytest.raises(ValueError, match="already exists"):
            configuration._check_name_exists("group")

        with pytest.raises(ValueError, match="already exists"):
            configuration._check_name_exists("group_2")
        
        with pytest.raises(ValueError, match="already exists"):
            configuration._check_name_exists("group_3")
        
        configuration._check_name_exists("group_4")

    def test_check_datasets_type_error(self, configuration):
        datasets_dict = [{"path_to_data": "table_name"}]
        datasets_list = [["path", "to", "data"], ["more", "data"]]
        datasets_correct = ["path_to_data", ("file_path", "table_name")]

        with pytest.raises(
            TypeError,
            match="datasets must be a list containing strings and/or tuples "
        ):
            configuration._check_datasets_type(datasets_dict)
            
        with pytest.raises(
            TypeError,
            match="datasets must be a list containing strings and/or tuples "
        ):
            configuration._check_datasets_type(datasets_list)

        configuration._check_datasets_type(datasets_correct)
