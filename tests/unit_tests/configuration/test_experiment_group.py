import textwrap

import pytest

from brisk.configuration.experiment_group import ExperimentGroup
from brisk.configuration import project

@pytest.fixture 
def valid_group(mock_regression_project):
    """Create a valid experiment group"""
    return ExperimentGroup(
        name="test_group",
        datasets=["test.csv"],
        algorithms=["linear", "ridge"],
        description="This is a test description"
    )


@pytest.fixture
def valid_group_two_datasets(mock_regression_project):
    """Create a valid experiment group"""
    return ExperimentGroup(
        name="test_group",
        datasets=["test.csv", "another_dataset.csv"],
        algorithms=["linear", "ridge"],
        description="This is a long description that should be wrapped over multiple lines since nobody wants to scroll across the screen to read this useless message."
    )


class TestExperimentGroup:
    def test_valid_creation(self, valid_group):
        """Test creation with valid parameters"""  
        assert valid_group.name == "test_group"
        assert valid_group.datasets == ["test.csv"]
        assert valid_group.algorithms == ["linear", "ridge"]

    def test_invalid_name(self, mock_regression_project):
        """Test creation with invalid name"""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name="", datasets=["test.csv"])
        
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name="   ", datasets=["test.csv"])

        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name=None, datasets=["test.csv"])

        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name=1, datasets=["test.csv"])

    def test_missing_dataset(self, mock_regression_project):
        """Test creation with non-existent dataset"""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            ExperimentGroup(
                name="test",
                datasets=["nonexistent.csv"]
            )

    def test_invalid_algorithm_config(self, mock_regression_project):
        """Test creation with invalid algorithm configuration"""
        with pytest.raises(
            ValueError, 
            match="Algorithm config contains algorithms not in the list of algorithms:"
        ):
            ExperimentGroup(
                name="test",
                datasets=["test.csv"],
                algorithms=["linear"],
                algorithm_config={"ridge": {"alpha": 1.0}}
            )

    def test_invalid_data_config(self, mock_regression_project):
        """Test creation with invalid data configuration"""
        with pytest.raises(ValueError, match="Invalid DataManager parameters"):
            ExperimentGroup(
                name="test",
                datasets=["test.csv"],
                data_config={"invalid_param": 1.0}
            )

    def test_dataset_paths(self, valid_group_two_datasets, mock_regression_project):
        """Test dataset_paths property"""
        expected_paths = [
            mock_regression_project / 'datasets' / 'test.csv',
            mock_regression_project / 'datasets' / 'another_dataset.csv'
        ]
        actual_paths  = [
            path for path, _ in valid_group_two_datasets.dataset_paths
        ]
        assert actual_paths == expected_paths

    def test_dataset_paths_exist(self, valid_group_two_datasets):
        """Test that dataset_paths point to existing files"""
        for path, _ in valid_group_two_datasets.dataset_paths:
            assert path.exists()

    @pytest.mark.parametrize("data_config", [
        {"test_size": 0.2},
        {"split_method": "kfold", "n_splits": 5},
        {"scale_method": "standard"},
        None
    ])
    def test_valid_data_configs(self, mock_regression_project, data_config):
        """Test various valid data configurations"""
        group = ExperimentGroup(
            name="test",
            datasets=["test.csv"],
            data_config=data_config
        )
        assert group.data_config == data_config

    def test_missing_datasets(self, mock_regression_project):
        """Test creation with missing dataset"""
        with pytest.raises(ValueError, match="At least one dataset must be specified"):
            ExperimentGroup(
                name="test_group",
                datasets=[],
                algorithms=["linear", "ridge"]
            )

    def test_short_description(self, valid_group):
        """Test short description is not modified"""
        assert valid_group.description == "This is a test description"

    def test_long_description(self, valid_group_two_datasets):
        """Test long description is wrapped"""
        expected_string = textwrap.dedent("""
        This is a long description that should be wrapped over
        multiple lines since nobody wants to scroll across the
        screen to read this useless message.
        """).strip()
        assert valid_group_two_datasets.description == expected_string

    def test_invalid_description(self, mock_regression_project):
        """Test invalid description raises ValueError"""
        with pytest.raises(ValueError, match="Description must be a string"):
            ExperimentGroup(
                name="test",
                datasets=["test.csv"],
                description=1
            )

    def test_invalid_workflow_args(self, mock_regression_project):
        with pytest.raises(ValueError, match="workflow_args must be a dict"):
            ExperimentGroup(
                name="test_invalid_args",
                datasets=["test.csv"],
                workflow_args=["arg1"]
            )

def test_project_root_not_found(tmp_path, monkeypatch):
    """Test FileNotFoundError when .briskconfig is not found"""
    monkeypatch.chdir(tmp_path)
    project.find_project_root.cache_clear()  # Clear before this specific test
    
    with pytest.raises(FileNotFoundError, match="Could not find .briskconfig"):
        project.find_project_root()

def test_nested_project_root(tmp_path, monkeypatch):
    """Test finding .briskconfig in parent directory"""
    # Create project structure
    project_root = tmp_path / "project"
    nested_dir = project_root / "nested" / "deep" / "directory"
    nested_dir.mkdir(parents=True)
    
    # Create .briskconfig in project root
    (project_root / '.briskconfig').touch()
    
    # Create datasets directory
    datasets_dir = project_root / 'datasets'
    datasets_dir.mkdir()
    (datasets_dir / 'test.csv').touch()
    
    # Change working directory to nested directory
    monkeypatch.chdir(nested_dir)
    project.find_project_root.cache_clear()  # Clear before this specific test
    
    group = ExperimentGroup(
        name="test_group",
        datasets=["test.csv"],
        algorithms=["linear"]
    )
    path, _ = group.dataset_paths[0]
    assert path == project_root / 'datasets' / 'test.csv'
