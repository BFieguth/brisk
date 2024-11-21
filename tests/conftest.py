import pytest
from pathlib import Path

from brisk.utility.utility import find_project_root

@pytest.fixture
def mock_project_root(tmp_path, monkeypatch):
    """Create a standardized temporary project structure for testing.
    
    Creates:
    - .briskconfig file
    - datasets directory with sample files
    - data.py with BASE_DATA_MANAGER
    
    Args:
        tmp_path: pytest fixture providing temporary directory
        monkeypatch: pytest fixture for modifying environment
        
    Returns:
        Path: Path to the mock project root directory
    """
    # Create base project structure
    (tmp_path / '.briskconfig').touch()
    
    # Create and populate datasets directory
    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()
    (datasets_dir / 'data.csv').touch()
    (datasets_dir / 'data_OLD.csv').touch()
    (datasets_dir / 'test.csv').touch()
    (datasets_dir / 'another_dataset.csv').touch()
    
    # Create data.py with BASE_DATA_MANAGER
    data_py = """
from brisk.data.DataManager import DataManager
BASE_DATA_MANAGER = DataManager()
"""
    (tmp_path / 'data.py').write_text(data_py)
    
    # Change working directory to project root
    monkeypatch.chdir(tmp_path)
    
    return tmp_path
    
    
@pytest.fixture(autouse=True)
def reset_project_root_cache():
    """Clear the project root cache after each test"""
    yield
    find_project_root.cache_clear()
    