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

    sample_data = {
        'data.csv': """x,y,target
1.0,2.0,0
2.0,3.0,1
3.0,4.0,0
4.0,5.0,1
5.0,6.0,0""",
        
        'data_OLD.csv': """feature1,feature2,label
0.1,0.2,A
0.3,0.4,B
0.5,0.6,A
0.7,0.8,B""",
        
        'test.csv': """col1,col2,col3
1,2,3
4,5,6
7,8,9""",
        
        'another_dataset.csv': """value,category,result
10.0,A,positive
20.0,B,negative
30.0,A,positive
40.0,B,negative"""
    }
    
    # Write data to CSV files
    for filename, content in sample_data.items():
        (datasets_dir / filename).write_text(content)

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
    