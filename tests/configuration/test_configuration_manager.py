import pytest
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
import numpy as np
import collections
import importlib
import pathlib

from brisk.configuration.configuration_manager import ConfigurationManager
from brisk.configuration.ExperimentGroup import ExperimentGroup
from brisk.data.DataManager import DataManager
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper

class TestConfigurationManager:
    def test_initialization(self, mock_regression_project):
        """Test basic initialization of ConfigurationManager."""
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

    def test_missing_data_file(self, mock_regression_project):
        """Test error handling for missing data file."""
        data_path = mock_regression_project / 'data.py'
        data_path.unlink()

        group = ExperimentGroup(
            name="test_group",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(FileNotFoundError, match="Data file not found:"):
            ConfigurationManager([group])

    def test_invalid_data_file(self, mock_regression_project):
        """Test error handling for data.py without BASE_DATA_MANAGER."""
        data_file = mock_regression_project / 'data.py'
        data_file.unlink()
        # Incorrect data.py file
        data_file.write_text("""
from brisk.data.DataManager import DataManager
data_manager = DataManager()
"""
        )

        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="BASE_DATA_MANAGER not found"):
            ConfigurationManager([group])

    def test_unloadable_data_file(self, mock_regression_project, monkeypatch):
        """Test error handling for invalid data.py file."""
        original_spec_from_file_location = importlib.util.spec_from_file_location
        
        def mock_spec_from_file_location(name, location, *args, **kwargs):
            if name == 'data':  # If module is 'data.py', return None
                return None
            return original_spec_from_file_location(
                name, location, *args, **kwargs
                )
            
        monkeypatch.setattr(
            'importlib.util.spec_from_file_location', 
            mock_spec_from_file_location
            )

        group = ExperimentGroup(
            name="test_group",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="Failed to load data module"):
            ConfigurationManager([group])          

    def test_missing_algorithm_file(self, mock_regression_project):
        """Test error handling for missing algorithms.py."""
        algorithm_file = mock_regression_project / 'algorithms.py'
        algorithm_file.unlink()

        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(FileNotFoundError, match="Algorithm config file not found"):
            ConfigurationManager([group])

    def test_invalid_algorithm_file(self, mock_regression_project):
        """Test error handling for algorithms.py without ALGORITHM_CONFIG."""
        algorithm_file = mock_regression_project / 'algorithms.py'
        algorithm_file.unlink()
        algorithm_file.write_text("""
# Missing ALGORITHM_CONFIG
"""
        )

        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="ALGORITHM_CONFIG not found"):
            ConfigurationManager([group])
      
    def test_unloadable_algorithm_file(self, mock_regression_project, monkeypatch):
        """Test error handling for invalid data.py file."""
        original_spec_from_file_location = importlib.util.spec_from_file_location
        
        def mock_spec_from_file_location(name, location, *args, **kwargs):
            if name == 'algorithms':  # If module is 'algorithms.py', return None
                return None
            return original_spec_from_file_location( # pragma: no cover
                name, location, *args, **kwargs
                )
            
        monkeypatch.setattr(
            'importlib.util.spec_from_file_location', 
            mock_spec_from_file_location
            )

        group = ExperimentGroup(
            name="test_group",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="Failed to load algorithms module"):
            ConfigurationManager([group])

    def test_get_base_params(self, mock_regression_project):
        """Test the _get_base_params method of ConfigurationManager."""
        group = ExperimentGroup(
            name="test_group",
            datasets=["data.csv"],
            algorithms=["linear"]
        )

        manager = ConfigurationManager([group])
        base_params = manager._get_base_params()
        
        expected_params = {
            'test_size': 0.2,
            'n_splits': 5,
            'split_method': 'shuffle',
            'group_column': None, 
            'stratified': False,
            'random_state': 42,
            'scale_method': None,
            'categorical_features': [] 
        }
        
        assert base_params == expected_params

    def test_data_manager_reuse(self, mock_regression_project):
        """Test that DataManagers are reused for matching configurations."""
        # monkeypatch.chdir(mock_project_files)
        groups = [
            ExperimentGroup(
                name="group1",
                datasets=["data.csv"],
                algorithms=["linear"]
            ),
            ExperimentGroup(
                name="group2",
                datasets=["data2.csv"],
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

    def test_experiment_creation(self, mock_regression_project):
        """Test creation of experiments from groups."""      
        groups = [
            ExperimentGroup(
                name="single",
                datasets=["data.csv"],
                algorithms=["linear"]
            ),
            ExperimentGroup(
                name="multiple",
                datasets=["data.csv", "data2.csv"],
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
            
    def test_create_logfile(self, mock_regression_project):
        """Test the _create_logfile method of ConfigurationManager."""
        group1 = ExperimentGroup(
            name="group1",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        group2 = ExperimentGroup(
            name="group2",
            datasets=["data.csv"],
            algorithms=["ridge"],
            algorithm_config={"ridge": {"alpha": 0.5}}
        )
        
        manager = ConfigurationManager([group1, group2])
        manager._create_logfile()

        expected_logfile_content = """
## Default Algorithm Configuration
### Linear Regression (`linear`)

- **Algorithm Class**: `LinearRegression`

**Default Parameters:**
```python
{}
```

**Hyperparameter Grid:**
```python
{}
```

### Ridge Regression (`ridge`)

- **Algorithm Class**: `Ridge`

**Default Parameters:**
```python
'max_iter': 10000,
```

**Hyperparameter Grid:**
```python
'alpha': [0.1, 0.5, 1.0],
```

### Elastic Net Regression (`elasticnet`)

- **Algorithm Class**: `ElasticNet`

**Default Parameters:**
```python
'alpha': 0.1,
'max_iter': 10000,
```

**Hyperparameter Grid:**
```python
'alpha': [0.1, 0.2, 0.5],
'l1_ratio': [0.1, 0.5, 1.0],
```

## Experiment Group: group1

### DataManager Configuration
```python
DataManager Configuration:
test_size: 0.2
n_splits: 5
split_method: shuffle
stratified: False
random_state: 42
```

### Datasets
#### data.csv
Features:
```python
Categorical: []
Continuous: ['x', 'y']
```

## Experiment Group: group2

### Algorithm Configurations
```python
'ridge': {'alpha': 0.5},
```

### DataManager Configuration
```python
DataManager Configuration:
test_size: 0.2
n_splits: 5
split_method: shuffle
stratified: False
random_state: 42
```

### Datasets
#### data.csv
Features:
```python
Categorical: []
Continuous: ['x', 'y']
```
"""
        # Strip whitespace from both files
        assert manager.logfile.strip() == expected_logfile_content.strip()

    def test_get_output_structure(self, mock_regression_project, tmp_path):
        """Test the _get_output_structure method of ConfigurationManager."""
        group1 = ExperimentGroup(
            name="group1",
            datasets=["data.csv"],
            algorithms=["linear"],
        )
        
        group2 = ExperimentGroup(
            name="group2",
            datasets=["data.csv"],
            algorithms=["ridge"],
        )
        
        manager = ConfigurationManager([group1, group2])
        output_structure = manager._get_output_structure()
        
        expected_output_structure = {
            "group1": {
                "data": (str(pathlib.Path(tmp_path / "datasets/data.csv")), "group1")
            },
            "group2": {
                "data": (str(pathlib.Path(tmp_path / "datasets/data.csv")), "group2")
            }
        }
        
        assert output_structure == expected_output_structure
