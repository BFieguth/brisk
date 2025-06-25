import pytest

from brisk.configuration import project
from brisk.configuration.experiment_group import ExperimentGroup

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
