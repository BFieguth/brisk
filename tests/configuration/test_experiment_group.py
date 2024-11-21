import pytest
from pathlib import Path

from brisk.configuration.ExperimentGroup import ExperimentGroup

@pytest.fixture
def mock_project_root(tmp_path, monkeypatch):
    """Create a temporary project structure"""
    (tmp_path / '.briskconfig').touch()
    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()
    (datasets_dir / 'test.csv').touch()
    (datasets_dir / 'another_dataset.csv').touch()
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def valid_group(mock_project_root):
    """Create a valid experiment group"""
    return ExperimentGroup(
        name="test_group",
        datasets=["test.csv"],
        algorithms=["linear", "ridge"]
    )


@pytest.fixture
def valid_group_two_datasets(mock_project_root):
    """Create a valid experiment group"""
    return ExperimentGroup(
        name="test_group",
        datasets=["test.csv", "another_dataset.csv"],
        algorithms=["linear", "ridge"]
    )


@pytest.fixture
def missing_dataset_group(mock_project_root):
    """Create a valid experiment group"""
    return ExperimentGroup(
        name="test_group",
        datasets=[],
        algorithms=["linear", "ridge"]
    )


class TestExperimentGroup:
    def test_valid_creation(self, valid_group):
        """Test creation with valid parameters"""
        assert valid_group.name == "test_group"
        assert valid_group.datasets == ["test.csv"]
        assert valid_group.algorithms == ["linear", "ridge"]

    def test_invalid_name(self, mock_project_root):
        """Test creation with invalid name"""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name="", datasets=["test.csv"])
        
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name="   ", datasets=["test.csv"])

        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name=None, datasets=["test.csv"])

        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name=1, datasets=["test.csv"])

    def test_missing_dataset(self, mock_project_root):
        """Test creation with non-existent dataset"""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            ExperimentGroup(
                name="test",
                datasets=["nonexistent.csv"]
            )

    def test_invalid_algorithm_config(self, mock_project_root):
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

    def test_invalid_data_config(self, mock_project_root):
        """Test creation with invalid data configuration"""
        with pytest.raises(ValueError, match="Invalid DataManager parameters"):
            ExperimentGroup(
                name="test",
                datasets=["test.csv"],
                data_config={"invalid_param": 1.0}
            )

    def test_dataset_paths(self, valid_group_two_datasets, mock_project_root):
        """Test dataset_paths property"""
        expected_paths = [
            mock_project_root / 'datasets' / 'test.csv',
            mock_project_root / 'datasets' / 'another_dataset.csv'
        ]
        assert valid_group_two_datasets.dataset_paths == expected_paths

    def test_dataset_paths_exist(self, valid_group_two_datasets):
        """Test that dataset_paths point to existing files"""
        for path in valid_group_two_datasets.dataset_paths:
            assert path.exists()

    @pytest.mark.parametrize("data_config", [
        {"test_size": 0.2},
        {"split_method": "kfold", "n_splits": 5},
        {"scale_method": "standard"},
        None
    ])
    def test_valid_data_configs(self, mock_project_root, data_config):
        """Test various valid data configurations"""
        group = ExperimentGroup(
            name="test",
            datasets=["test.csv"],
            data_config=data_config
        )
        assert group.data_config == data_config

    def test_missing_datasets(self, mock_project_root):
        """Test creation with missing dataset"""
        with pytest.raises(ValueError, match="At least one dataset must be specified"):
            ExperimentGroup(
                name="test_group",
                datasets=[],
                algorithms=["linear", "ridge"]
            )

def test_project_root_not_found(tmp_path, monkeypatch):
    """Test FileNotFoundError when .briskconfig is not found"""
    # Change working directory to a temporary directory without .briskconfig
    monkeypatch.chdir(tmp_path)
    
    with pytest.raises(FileNotFoundError, match="Could not find .briskconfig"):
        ExperimentGroup(
            name="test_group",
            datasets=["test.csv"],
            algorithms=["linear"]
        )

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
    
    # Should find .briskconfig in parent
    group = ExperimentGroup(
        name="test_group",
        datasets=["test.csv"],
        algorithms=["linear"]
    )
    
    assert group.dataset_paths[0] == project_root / 'datasets' / 'test.csv'