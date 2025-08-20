import textwrap

import pytest

from brisk.configuration.experiment_group import ExperimentGroup

@pytest.fixture 
def valid_group(mock_brisk_project):
    """Create a valid experiment group"""
    return ExperimentGroup(
        name="test_group",
        datasets=["regression.csv"],
        algorithms=["linear", "ridge"],
        description="This is a test description"
    )


@pytest.fixture
def valid_group_two_datasets(mock_brisk_project):
    """Create a valid experiment group"""
    return ExperimentGroup(
        name="test_group",
        datasets=["regression.csv", "categorical.csv"],
        algorithms=["linear", "ridge"],
        description="This is a long description that should be wrapped over multiple lines since nobody wants to scroll across the screen to read this useless message."
    )


class TestExperimentGroup:
    def test_valid_creation(self, valid_group):
        """Test creation with valid parameters"""  
        assert valid_group.name == "test_group"
        assert valid_group.datasets == ["regression.csv"]
        assert valid_group.algorithms == ["linear", "ridge"]
        assert valid_group.algorithm_config == None
        assert valid_group.description == "This is a test description"
        assert valid_group.workflow_args == None

    def test_invalid_name(self, mock_brisk_project):
        """Test creation with invalid name"""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name="", datasets=["regression.csv"])
        
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name="   ", datasets=["regression.csv"])

        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name=None, datasets=["regression.csv"])

        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name=1, datasets=["regression.csv"])

        # Check valid names don't raise errors
        ExperimentGroup(name="test_group", datasets=["regression.csv"])
        ExperimentGroup(name="test group", datasets=["regression.csv"])
        ExperimentGroup(name="   test   ", datasets=["regression.csv"])

    def test_missing_dataset(self, mock_brisk_project):
        """Test creation with non-existent dataset"""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            ExperimentGroup(
                name="test",
                datasets=["nonexistent.csv"]
            )

    def test_missing_datasets(self, mock_brisk_project):
        """Test creation with missing dataset"""
        with pytest.raises(ValueError, match="At least one dataset must be specified"):
            ExperimentGroup(
                name="test_group",
                datasets=[],
                algorithms=["linear", "ridge"]
            )

    def test_dataset_paths(self, valid_group_two_datasets, mock_brisk_project):
        """Test dataset_paths property"""
        expected_paths = [
            mock_brisk_project / 'datasets' / 'regression.csv',
            mock_brisk_project / 'datasets' / 'categorical.csv'
        ]
        actual_paths  = [
            path for path, _ in valid_group_two_datasets.dataset_paths
        ]
        assert actual_paths == expected_paths

    def test_dataset_paths_exist(self, valid_group_two_datasets):
        """Test that dataset_paths point to existing files"""
        for path, _ in valid_group_two_datasets.dataset_paths:
            assert path.exists()

    def test_invalid_algorithm_config(self, mock_brisk_project):
        """Test creation with invalid algorithm configuration"""
        with pytest.raises(
            ValueError, 
            match="Algorithm config contains algorithms not in the list of algorithms:"
        ):
            ExperimentGroup(
                name="test",
                datasets=["regression.csv"],
                algorithms=["linear"],
                algorithm_config={"ridge": {"alpha": 1.0}}
            )

    def test_invalid_algorithm_config_nested(self, mock_brisk_project):
        """Test creation with invalid algorithm configuration"""
        with pytest.raises(
            ValueError, 
            match="Algorithm config contains algorithms not in the list of algorithms:"
        ):
            ExperimentGroup(
                name="test",
                datasets=["regression.csv"],
                algorithms=[["linear", "ridge"]],
                algorithm_config={"elasticnet": {"alpha": 1.0}}
            )

        # Check nested alorithms are found correctly
        ExperimentGroup(
            name="test",
            datasets=["regression.csv"],
            algorithms=[["linear", "ridge"]],
            algorithm_config={"ridge": {"alpha": 1.0}}
        )

    def test_invalid_data_config(self, mock_brisk_project):
        """Test creation with invalid data configuration"""
        with pytest.raises(ValueError, match="Invalid DataManager parameters"):
            ExperimentGroup(
                name="test",
                datasets=["regression.csv"],
                data_config={"invalid_param": 1.0}
            )

    @pytest.mark.parametrize("data_config", [
        {"test_size": 0.2},
        {"split_method": "kfold", "n_splits": 5},
        None
    ])
    def test_valid_data_configs(self, mock_brisk_project, data_config):
        """Test various valid data configurations"""
        group = ExperimentGroup(
            name="test",
            datasets=["regression.csv"],
            data_config=data_config
        )
        assert group.data_config == data_config

    def test_long_description(self, valid_group_two_datasets):
        """Test long description is wrapped"""
        expected_string = textwrap.dedent("""
        This is a long description that should be wrapped over
        multiple lines since nobody wants to scroll across the
        screen to read this useless message.
        """).strip()
        assert valid_group_two_datasets.description == expected_string

    def test_invalid_description(self, mock_brisk_project):
        """Test invalid description raises ValueError"""
        with pytest.raises(ValueError, match="Description must be a string"):
            ExperimentGroup(
                name="test",
                datasets=["regression.csv"],
                description=1
            )
        
        with pytest.raises(ValueError, match="Description must be a string"):
            ExperimentGroup(
                name="test",
                datasets=["regression.csv"],
                description=["A description", "that is not a string"]
            )

    def test_invalid_workflow_args(self, mock_brisk_project):
        with pytest.raises(ValueError, match="workflow_args must be a dict"):
            ExperimentGroup(
                name="test_invalid_args",
                datasets=["regression.csv"],
                workflow_args=["arg1"]
            )
