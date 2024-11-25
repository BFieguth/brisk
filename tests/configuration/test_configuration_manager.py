import pytest
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
import numpy as np
import collections
import importlib

from brisk.configuration.ConfigurationManager import ConfigurationManager
from brisk.configuration.ExperimentGroup import ExperimentGroup
from brisk.data.DataManager import DataManager
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper

@pytest.fixture
def mock_project_files(mock_project_root):
    """Create necessary project configuration files."""
    # Create data.py with BASE_DATA_MANAGER
    data_content = """
from brisk.data.DataManager import DataManager

BASE_DATA_MANAGER = DataManager(
    test_size=0.2,
    random_state=42
)
"""
    (mock_project_root / 'data.py').write_text(data_content)
    
    # Create algorithms.py with ALGORITHM_CONFIG
    algo_content = """
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper

ALGORITHM_CONFIG = [
    AlgorithmWrapper(
        name="linear",
        display_name="Linear Regression",
        algorithm_class=LinearRegression,
        default_params={},
        hyperparam_grid={}
    ),
    AlgorithmWrapper(
        name="ridge",
        display_name="Ridge Regression",
        algorithm_class=Ridge,
        default_params={"alpha": 1.0},
        hyperparam_grid={"alpha": [0.1, 1.0]}
    ),
    AlgorithmWrapper(
        name="elasticnet",
        display_name="Elastic Net",
        algorithm_class=ElasticNet,
        default_params={"alpha": 1.0, "l1_ratio": 0.5},
        hyperparam_grid={"alpha": [0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]}
    )
]
"""
    (mock_project_root / 'algorithms.py').write_text(algo_content)
    
    # Create test datasets
    datasets_dir = mock_project_root / 'datasets'
    datasets_dir.mkdir(exist_ok=True)
    (datasets_dir / 'data1.csv').touch()
    (datasets_dir / 'data2.csv').touch()
    
    return mock_project_root


@pytest.fixture
def mock_project_missing_data_file(mock_project_files):
    """Create project files but remove data.py."""
    data_file = mock_project_files / 'data.py'
    data_file.unlink()
    return mock_project_files


class TestConfigurationManager:
    def test_initialization(self, mock_project_files, monkeypatch):
        """Test basic initialization of ConfigurationManager."""
        monkeypatch.chdir(mock_project_files)
        
        group = ExperimentGroup(
            name="test_group",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        manager = ConfigurationManager([group])
        
        assert isinstance(manager.base_data_manager, DataManager)
        assert isinstance(manager.data_managers, dict)
        assert isinstance(manager.experiment_queue, collections.deque)
        assert len(manager.experiment_queue) == 1

    def test_data_manager_reuse(self, mock_project_files, monkeypatch):
        """Test that DataManagers are reused for matching configurations."""
        monkeypatch.chdir(mock_project_files)
        
        groups = [
            ExperimentGroup(
                name="group1",
                datasets=["data.csv"],
                algorithms=["linear"]
            ),
            ExperimentGroup(
                name="group2",
                datasets=["data_OLD.csv"],
                algorithms=["ridge"],
                data_config={"test_size": 0.3}
            ),
            ExperimentGroup(
                name="group3",
                datasets=["test.csv"],
                algorithms=["elasticnet"]
            )
        ]
        
        manager = ConfigurationManager(groups)
        
        # group1 and group3 should share the base DataManager
        assert manager.data_managers["group1"] is manager.data_managers["group3"]
        # group2 should have its own DataManager
        assert manager.data_managers["group2"] is not manager.data_managers["group1"]

    def test_missing_algorithm_file(self, mock_project_root, monkeypatch):
        """Test error handling for missing algorithms.py."""
        monkeypatch.chdir(mock_project_root)
        
        datasets_dir = mock_project_root / 'datasets'
        (datasets_dir / 'data1.csv').touch()
        (mock_project_root / 'data.py').write_text("""
from brisk.data.DataManager import DataManager
BASE_DATA_MANAGER = DataManager()
""")
        
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(FileNotFoundError, match="Algorithm config file not found"):
            ConfigurationManager([group])

    def test_invalid_data_file(self, mock_project_root, monkeypatch):
        """Test error handling for data.py without BASE_DATA_MANAGER."""
        monkeypatch.chdir(mock_project_root)

        # Create datasets directory and sample data file
        datasets_dir = mock_project_root / 'datasets'
        datasets_dir.mkdir(exist_ok=True)
        (datasets_dir / 'data.csv').write_text("col1,col2\n1,2\n3,4")

        # Create required algorithms.py with minimal config
        (mock_project_root / 'algorithms.py').write_text("""
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper
from sklearn.linear_model import LinearRegression
ALGORITHM_CONFIG = [
    AlgorithmWrapper(
        name="linear", 
        display_name="Linear Regression", 
        algorithm_class=LinearRegression
    )
]
""")

        # Create invalid data.py (missing BASE_DATA_MANAGER)
        (mock_project_root / 'data.py').write_text("""
from brisk.data.DataManager import DataManager
# Missing BASE_DATA_MANAGER
""")
        
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="BASE_DATA_MANAGER not found"):
            ConfigurationManager([group])

    def test_invalid_algorithm_file(self, mock_project_root, monkeypatch):
        """Test error handling for algorithms.py without ALGORITHM_CONFIG."""
        monkeypatch.chdir(mock_project_root)
        
        datasets_dir = mock_project_root / 'datasets'
        (datasets_dir / 'data1.csv').touch()
        
        (mock_project_root / 'data.py').write_text("""
from brisk.data.DataManager import DataManager
BASE_DATA_MANAGER = DataManager()
""")

        (mock_project_root / 'algorithms.py').write_text("""
from sklearn.linear_model import LinearRegression
# Missing ALGORITHM_CONFIG
""")
        
        group = ExperimentGroup(
            name="test",
            datasets=["data1.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="ALGORITHM_CONFIG not found"):
            ConfigurationManager([group])

    def test_experiment_creation(self, mock_project_files, monkeypatch):
        """Test creation of experiments from groups."""
        monkeypatch.chdir(mock_project_files)
        
        groups = [
            ExperimentGroup(
                name="single",
                datasets=["data.csv"],
                algorithms=["linear"]
            ),
            ExperimentGroup(
                name="multiple",
                datasets=["data.csv", "data_OLD.csv"],
                algorithms=[["ridge", "elasticnet"]],
                algorithm_config={
                    "ridge": {"alpha": np.logspace(-3, 3, 5)},
                }
            )
        ]
        
        manager = ConfigurationManager(groups)
        
        assert len(manager.experiment_queue) == 3
        
        # Check experiment configurations
        single_exp = next(
            exp for exp in manager.experiment_queue 
            if len(exp.algorithms) == 1
        )
        assert isinstance(single_exp.algorithms["model"], AlgorithmWrapper)
        assert single_exp.algorithms["model"].algorithm_class == LinearRegression

        multi_exps = [
            exp for exp in manager.experiment_queue 
            if len(exp.algorithms) == 2
        ]
        assert len(multi_exps) == 2
        for exp in multi_exps:
            assert isinstance(exp.algorithms["model1"], AlgorithmWrapper)
            assert isinstance(exp.algorithms["model2"], AlgorithmWrapper)
            assert exp.algorithms["model1"].algorithm_class == Ridge
            assert exp.algorithms["model2"].algorithm_class == ElasticNet
            
    def test_missing_data_file(self, mock_project_missing_data_file, monkeypatch):
        """Test error handling for missing data file."""
        monkeypatch.chdir(mock_project_missing_data_file)

        group = ExperimentGroup(
            name="test_group",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(FileNotFoundError, match="Data file not found:"):
            ConfigurationManager([group])

    def test_unloadable_data_file(self, mock_project_files, monkeypatch):
        """Test error handling for invalid data.py file."""
        monkeypatch.chdir(mock_project_files)
        
        original_spec_from_file_location = importlib.util.spec_from_file_location
        
        def mock_spec_from_file_location(name, location, *args, **kwargs):
            if name == 'data':  # If module is 'data.py', return None
                return None
            return original_spec_from_file_location(name, location, *args, **kwargs)
            
        monkeypatch.setattr('importlib.util.spec_from_file_location', mock_spec_from_file_location)

        group = ExperimentGroup(
            name="test_group",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="Failed to load data module"):
            ConfigurationManager([group])
                
    def test_unloadable_algorithm_file(self, mock_project_files, monkeypatch):
        """Test error handling for invalid data.py file."""
        monkeypatch.chdir(mock_project_files)
        
        original_spec_from_file_location = importlib.util.spec_from_file_location
        
        def mock_spec_from_file_location(name, location, *args, **kwargs):
            if name == 'algorithms':  # If module is 'algorithms.py', return None
                return None
            return original_spec_from_file_location(name, location, *args, **kwargs) # pragma: no cover
            
        monkeypatch.setattr('importlib.util.spec_from_file_location', mock_spec_from_file_location)

        group = ExperimentGroup(
            name="test_group",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="Failed to load algorithms module"):
            ConfigurationManager([group])
                                