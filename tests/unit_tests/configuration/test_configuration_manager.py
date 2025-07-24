"""Unit tests for the ConfigurationManager class."""

import pytest
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
import numpy as np
import collections
import importlib
import pathlib
import textwrap

from brisk.configuration.configuration_manager import ConfigurationManager
from brisk.configuration.experiment_group import ExperimentGroup
from brisk.data.data_manager import DataManager
from brisk.configuration.algorithm_wrapper import AlgorithmWrapper, AlgorithmCollection

class TestConfigurationManager:
    """Unit tests for the ConfigurationManager class."""
    def test_initialization(self, mock_brisk_project):
        """Test basic initialization of ConfigurationManager."""
        group = ExperimentGroup(
            name="test_group",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        group2 = ExperimentGroup(
            name="test_group2", 
            datasets=["categorical.csv"],
            algorithms=["ridge"]
        )
        experiment_groups = [group, group2]
        manager = ConfigurationManager(experiment_groups, {})
        
        assert manager.experiment_groups == experiment_groups
        assert manager.categorical_features == {}
        assert manager.project_root == mock_brisk_project
        assert isinstance(manager.algorithm_config, AlgorithmCollection)
        assert isinstance(manager.base_data_manager, DataManager)
        assert isinstance(manager.data_managers, dict)
        assert [*manager.data_managers.keys()] == ["test_group", "test_group2"]
        assert isinstance(manager.experiment_queue, collections.deque)
        assert len(manager.experiment_queue) == 2

    def test_missing_data_file(self, mock_brisk_project):
        """Test error handling for missing data file."""
        data_path = mock_brisk_project / 'data.py'
        data_path.unlink()

        group = ExperimentGroup(
            name="test_group",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(FileNotFoundError, match="Data file not found:"):
            ConfigurationManager([group], {})

    def test_invalid_data_file(self, mock_brisk_project):
        """Test error handling for data.py without BASE_DATA_MANAGER."""
        data_file = mock_brisk_project / 'data.py'
        data_file.unlink()
        # Incorrect data.py file
        data_content = textwrap.dedent("""
            from brisk.data.data_manager import DataManager
            data_manager = DataManager()
        """).strip()
        data_file.write_text(data_content)

        group = ExperimentGroup(
            name="test",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="BASE_DATA_MANAGER not found"):
            ConfigurationManager([group], {})

    def test_unloadable_data_file(self, mock_brisk_project, monkeypatch):
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
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="Failed to load data module"):
            ConfigurationManager([group], {})      

    def test_two_data_managers(self, mock_brisk_project):
        """Test correct data manager is loaded if multiple are defined"""
        data_file = mock_brisk_project / "data.py"
        data_file.unlink()
        data_content = textwrap.dedent("""
            from brisk.data.data_manager import DataManager
            BASE_DATA_MANAGER = DataManager(
                test_size=0.2,
                n_splits=5,
                split_method="shuffle"
            )
            BASE_DATA_MANAGER = DataManager(
                test_size=0.3,
                n_splits=5,
                split_method="kfold"
            )
        """).strip()
        data_file.write_text(data_content)
        group = ExperimentGroup(
            name="test_group",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        with pytest.raises(
            ValueError, 
            match="BASE_DATA_MANAGER is defined multiple times in"
            ):
            manager = ConfigurationManager([group], {})

    def test_base_data_manager_wrong_class(self, mock_brisk_project):
        """Test error handling for invalid base data manager class"""
        data_file = mock_brisk_project / "data.py"
        data_file.unlink()
        data_content = textwrap.dedent("""
            from brisk.data.data_manager import DataManager
            BASE_DATA_MANAGER = 1
        """).strip()
        data_file.write_text(data_content)
        with pytest.raises(
            ValueError, 
            match="is not a valid DataManager instance"
            ):
            manager = ConfigurationManager([], {})
    
    def test_validate_single_data_manager(self,mock_brisk_project):
        """Test the _validate_single_data_manager method correct behavior."""
        group = ExperimentGroup(
            name="test_group",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        manager = ConfigurationManager([group], {})
        assert manager._validate_single_variable(mock_brisk_project / "data.py", "BASE_DATA_MANAGER") is None

    def test_validate_single_data_manager_two_definitions(
            self,
            mock_brisk_project
        ):
        """Test the _validate_single_data_manager method error handling."""
        data_file = mock_brisk_project / "data.py"
        data_file.unlink()
        data_content = textwrap.dedent("""
            from brisk.data.data_manager import DataManager
            BASE_DATA_MANAGER = DataManager(
                test_size=0.2,
                n_splits=5,
                split_method="shuffle"
            )
            BASE_DATA_MANAGER = DataManager(
                test_size=0.3,
                n_splits=5,
                split_method="kfold"
            )
        """).strip()
        data_file.write_text(data_content)

        with pytest.raises(
            ValueError, 
            match="BASE_DATA_MANAGER is defined multiple times in"
            ):
            manager = ConfigurationManager([], {})

    def test_validate_single_data_manager_invalid_syntax(
            self,
            mock_brisk_project
        ):
        """Test the _validate_single_data_manager method error handling."""
        data_file = mock_brisk_project / "data.py"
        data_file.unlink()
        data_content = textwrap.dedent("""
            from brisk.data.data_manager import DataManager
            BASE_DATA_MANAGER = DataManager(
                test_size=0.2
                n_splits=5
                split_method="shuffle"
            )
        """).strip()
        data_file.write_text(data_content)

        with pytest.raises(SyntaxError, match="invalid syntax"):
            manager = ConfigurationManager([], {})

    def test_missing_algorithm_file(self, mock_brisk_project):
        """Test error handling for missing algorithms.py."""
        algorithm_file = mock_brisk_project / 'algorithms.py'
        algorithm_file.unlink()
        group = ExperimentGroup(
            name="test",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(
            FileNotFoundError, 
            match="algorithms.py file not found:"
            ):
            ConfigurationManager([group], {})

    def test_missing_algorithm_config(self, mock_brisk_project):
        """Test error handling for algorithms.py without ALGORITHM_CONFIG."""
        algorithm_file = mock_brisk_project / 'algorithms.py'
        algorithm_file.unlink()
        algorithm_content = textwrap.dedent("""
            # Missing ALGORITHM_CONFIG
        """).strip()
        algorithm_file.write_text(algorithm_content)
        group = ExperimentGroup(
            name="test",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="ALGORITHM_CONFIG not found in"):
            ConfigurationManager([group], {})
      
    def test_invalid_algorithm_file(self, mock_brisk_project):
        """Test error handling for invalid algorithms.py file."""
        algorithm_file = mock_brisk_project / 'algorithms.py'
        algorithm_file.unlink()
        algorithm_content = textwrap.dedent("""
            from brisk.configuration.algorithm_wrapper import AlgorithmCollection
            algorithm_config = AlgorithmCollection()
        """).strip()
        algorithm_file.write_text(algorithm_content)

        with pytest.raises(
            ImportError, 
            match="ALGORITHM_CONFIG not found in"
            ):
            manager = ConfigurationManager([], {})

    def test_unloadable_algorithm_file(self, mock_brisk_project, monkeypatch):
        """Test error handling for invalid data.py file."""
        original_spec_from_file_location = importlib.util.spec_from_file_location

        def mock_spec_from_file_location(name, location, *args, **kwargs):
            if name == 'algorithms': # If module is 'algorithms.py', return None
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
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        
        with pytest.raises(ImportError, match="Failed to load algorithms module"):
            ConfigurationManager([group], {})

    def test_validate_single_algorithm_config(self, mock_brisk_project):
        """Test the ALGORITHM_CONFIG is an AlgorithmCollection."""
        group = ExperimentGroup(
            name="test_group",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        manager = ConfigurationManager([group], {})
        assert manager._validate_single_variable(
            mock_brisk_project / "algorithms.py", 
            "ALGORITHM_CONFIG"
            ) is None

    def test_validate_two_algorithm_configs(self, mock_brisk_project):
        """Test two ALGORITHM_CONFIGs are not allowed."""
        algorithm_file = mock_brisk_project / 'algorithms.py'
        algorithm_file.unlink()
        algorithm_content = textwrap.dedent("""
            from brisk.configuration.algorithm_wrapper import AlgorithmCollection
            ALGORITHM_CONFIG = AlgorithmCollection()
            ALGORITHM_CONFIG = AlgorithmCollection()
        """).strip()
        algorithm_file.write_text(algorithm_content)
        with pytest.raises(
            ValueError, 
            match="ALGORITHM_CONFIG is defined multiple times in"
            ):
            manager = ConfigurationManager([], {})

    def test_single_algorithm_config_invalid_syntax(
            self,
            mock_brisk_project
        ):
        """Test error handling for invalid ALGORITHM_CONFIG syntax."""
        algorithm_file = mock_brisk_project / 'algorithms.py'
        algorithm_file.unlink()
        algorithm_content = textwrap.dedent("""
            from brisk.configuration.algorithm_wrapper import AlgorithmCollection, AlgorithmWrapper
            from sklearn.linear_model import LinearRegression
            ALGORITHM_CONFIG = AlgorithmCollection(
                AlgorithmWrapper(
                    name="linear"
                    display_name="Linear Regression"
                    algorithm_class=LinearRegression
                )
            )
        """).strip()
        algorithm_file.write_text(algorithm_content)

        with pytest.raises(SyntaxError, match="invalid syntax"):
            manager = ConfigurationManager([], {})

    def test_get_base_params(self, mock_brisk_project):
        """Test the _get_base_params method of ConfigurationManager."""
        group = ExperimentGroup(
            name="test_group",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        manager = ConfigurationManager([group], {})
        base_params = manager._get_base_params()
        expected_params = {
            'test_size': 0.2,
            'n_splits': 5,
            'split_method': 'shuffle',
            'group_column': None, 
            'stratified': False,
            'random_state': 42,
            'scale_method': None,
            'feature_selection_method': None,
            'feature_selection_estimator': None,
            'problem_type': 'classification',
            'n_features_to_select': 5,
            'feature_selection_cv': 3,
            'algorithm_config': None
        }
        assert base_params == expected_params
        print("Actual base_params:", base_params)
        print("Expected params:", expected_params)
    def test_data_config_does_not_change_base(self, mock_brisk_project):
        """Test passing data_config arg does not change the base data manager"""
        group = ExperimentGroup(
            name="test_group",
            datasets=["regression.csv"],
            algorithms=["linear"],
            data_config={
                "test_size": 0.4,
                "split_method": "kfold"
            }
        )
        manager = ConfigurationManager([group], {})
        base_params = manager._get_base_params()
        expected_params = {
            'test_size': 0.2,
            'n_splits': 5,
            'split_method': 'shuffle',
            'group_column': None, 
            'stratified': False,
            'random_state': 42,
            'scale_method': None,
            'feature_selection_method': None,
            'feature_selection_estimator': None,
            'problem_type': 'classification',
            'n_features_to_select': 5,
            'feature_selection_cv': 3,
            'algorithm_config': None
        }
        assert base_params == expected_params

    def test_data_manager_reuse(self, mock_brisk_project):
        """Test that DataManagers are reused for matching configurations."""
        groups = [
            ExperimentGroup(
                name="group1",
                datasets=["regression.csv"],
                algorithms=["linear"]
            ),
            ExperimentGroup(
                name="group2",
                datasets=["categorical.csv"],
                algorithms=["ridge"],
                data_config={"test_size": 0.3}
            ),
            ExperimentGroup(
                name="group3",
                datasets=["regression.csv"],
                algorithms=["elasticnet"]
            )
        ]
        manager = ConfigurationManager(groups, {})
        for data_manager in manager.data_managers.values():
            assert isinstance(data_manager, DataManager)
        assert len(manager.data_managers) == 3
        # group1 and group3 should share the base DataManager
        assert manager.data_managers["group1"] is manager.data_managers["group3"]
        # group2 should have its own DataManager
        assert manager.data_managers["group2"] is not manager.data_managers["group1"]

    def test_experiment_creation(self, mock_brisk_project):
        """Test creation of experiments from groups."""      
        groups = [
            ExperimentGroup(
                name="single",
                datasets=["regression.csv"],
                algorithms=["linear"]
            ),
            ExperimentGroup(
                name="multiple",
                datasets=["regression.csv", "categorical.csv"],
                algorithms=[["ridge", "elasticnet"]],
                algorithm_config={
                    "ridge": {"alpha": np.logspace(-3, 3, 5)},
                }
            )
        ]
        
        manager = ConfigurationManager(groups, {})
        
        assert len(manager.experiment_queue) == 3
        assert isinstance(manager.experiment_queue, collections.deque)
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
            assert isinstance(exp.algorithms["model"], AlgorithmWrapper)
            assert isinstance(exp.algorithms["model2"], AlgorithmWrapper)
            assert exp.algorithms["model"].algorithm_class == Ridge
            assert exp.algorithms["model2"].algorithm_class == ElasticNet

    def test_correct_experiment_queue_length(self, mock_brisk_project):
        """Test the correct length of the experiment queue."""
        groups = [
            ExperimentGroup(
                name="group1",
                datasets=["regression.csv", "categorical.csv"],
                algorithms=["linear", "ridge", "elasticnet"]
            ),
            ExperimentGroup(
                name="group2",
                datasets=["regression.csv"],
                algorithms=[["ridge", "elasticnet"], ["linear", "ridge"]]
            ),
            ExperimentGroup(
                name="group3",
                datasets=["regression.csv"],
                algorithms=["linear", "ridge"]
            )
        ]
        manager = ConfigurationManager(groups, {})
        assert len(manager.experiment_queue) == 10

    def test_create_logfile(self, mock_brisk_project):
        """Test the _create_logfile method of ConfigurationManager."""
        group1 = ExperimentGroup(
            name="group1",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )

        group2 = ExperimentGroup(
            name="group2",
            datasets=["regression.csv"],
            algorithms=["ridge"],
            algorithm_config={"ridge": {"alpha": 0.5}}
        )
        
        manager = ConfigurationManager([group1, group2], {})
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

### Random Forest (`rf`)

- **Algorithm Class**: `RandomForestRegressor`

**Default Parameters:**
```python
'n_jobs': 1,
```

**Hyperparameter Grid:**
```python
'n_estimators': [20, 40, 60, 80, 100, 120, 140],
```

### Random Forest Classifier (`rf_classifier`)

- **Algorithm Class**: `RandomForestClassifier`

**Default Parameters:**
```python
'min_samples_split': 10,
```

**Hyperparameter Grid:**
```python
'n_estimators': [20, 40, 60, 80, 100, 120, 140],
'criterion': ['friedman_mse', 'absolute_error', 'poisson', 'squared_error'],
'max_depth': [5, 10, 15, 20, None],
```

## Experiment Group: group1
#### Description: 

### DataManager Configuration
```python
DataManager Configuration:
test_size: 0.2
n_splits: 5
split_method: shuffle
stratified: False
random_state: 42
problem_type: classification
n_features_to_select: 5
feature_selection_cv: 3
```

### Datasets
#### regression.csv
Features:
```python
Categorical: []
Continuous: ['x', 'y']
```

## Experiment Group: group2
#### Description: 

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
problem_type: classification
n_features_to_select: 5
feature_selection_cv: 3
```

### Datasets
#### regression.csv
Features:
```python
Categorical: []
Continuous: ['x', 'y']
```
"""
        # Strip whitespace from both files
        print("ACTUAL LOGFILE CONTENT:")
        print(manager.logfile)
        print("EXPECTED LOGFILE CONTENT:")
        print(expected_logfile_content)
        assert manager.logfile.strip() == expected_logfile_content.strip()

    def test_create_logfile_with_all_args(self, mock_brisk_project):
        """Test the _create_logfile method of ConfigurationManager."""
        group1 = ExperimentGroup(
            name="group1",
            datasets=["regression.csv", "categorical.csv"],
            data_config={
                "test_size": 0.3,
                "n_splits": 3
            },
            algorithms=["linear", "ridge", "elasticnet"],
            algorithm_config={
                "ridge": {"alpha": [0.1, 0.2, 0.5, 0.7, 0.9]},
                "elasticnet": {
                    "alpha": [0.1, 0.2, 0.6, 0.75, 0.95],
                    "l1_ratio": [0.01, 0.05, 0.1, 0.5, 0.8]
                }
            },
            description="This is a test description of group1",
            workflow_args={
                "kfold": 2,
                "metrics": ["MAE", "R2"]
            }
        )

        group2 = ExperimentGroup(
            name="group2",
            datasets=["group.csv"],
            data_config={
                "test_size": 0.1,
                "split_method": "shuffle",
                "group_column": "group"
            },
            algorithms=["ridge", "elasticnet"],
            algorithm_config={"ridge": {"alpha": [0.5, 0.3, 0.1]}},
            description="This describes group2 in great detail",
            workflow_args={
                "kfold": 3,
                "metrics": ["MSE", "CCC"]
            }
        )
        
        manager = ConfigurationManager([group1, group2], {
            "categorical.csv": ["category"]
        })
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

### Random Forest (`rf`)

- **Algorithm Class**: `RandomForestRegressor`

**Default Parameters:**
```python
'n_jobs': 1,
```

**Hyperparameter Grid:**
```python
'n_estimators': [20, 40, 60, 80, 100, 120, 140],
```

### Random Forest Classifier (`rf_classifier`)

- **Algorithm Class**: `RandomForestClassifier`

**Default Parameters:**
```python
'min_samples_split': 10,
```

**Hyperparameter Grid:**
```python
'n_estimators': [20, 40, 60, 80, 100, 120, 140],
'criterion': ['friedman_mse', 'absolute_error', 'poisson', 'squared_error'],
'max_depth': [5, 10, 15, 20, None],
```

## Experiment Group: group1
#### Description: This is a test description of group1

### Algorithm Configurations
```python
'ridge': {'alpha': [0.1, 0.2, 0.5, 0.7, 0.9]},
'elasticnet': {'alpha': [0.1, 0.2, 0.6, 0.75, 0.95], 'l1_ratio': [0.01, 0.05, 0.1, 0.5, 0.8]},
```

### DataManager Configuration
```python
DataManager Configuration:
test_size: 0.3
n_splits: 3
split_method: shuffle
stratified: False
random_state: 42
problem_type: classification
n_features_to_select: 5
feature_selection_cv: 3
```

### Datasets
#### regression.csv
Features:
```python
Categorical: []
Continuous: ['x', 'y']
```

#### categorical.csv
Features:
```python
Categorical: ['category']
Continuous: ['value']
```

## Experiment Group: group2
#### Description: This describes group2 in great detail

### Algorithm Configurations
```python
'ridge': {'alpha': [0.5, 0.3, 0.1]},
```

### DataManager Configuration
```python
DataManager Configuration:
test_size: 0.1
n_splits: 5
split_method: shuffle
group_column: group
stratified: False
random_state: 42
problem_type: classification
n_features_to_select: 5
feature_selection_cv: 3
```

### Datasets
#### group.csv
Features:
```python
Categorical: []
Continuous: ['x', 'y']
```
"""
        # Strip whitespace from both files
        print("ACTUAL LOGFILE CONTENT:")
        print(manager.logfile)
        print("EXPECTED LOGFILE CONTENT:")
        print(expected_logfile_content)
        assert manager.logfile.strip() == expected_logfile_content.strip()

    def test_get_output_structure(self, mock_brisk_project, tmp_path):
        """Test the _get_output_structure method of ConfigurationManager."""
        group1 = ExperimentGroup(
            name="group1",
            datasets=["regression.csv"],
            algorithms=["linear"],
        )
        
        group2 = ExperimentGroup(
            name="group2",
            datasets=["regression.csv"],
            algorithms=["ridge"],
        )
        
        manager = ConfigurationManager([group1, group2], {})
        output_structure = manager._get_output_structure()
        
        expected_output_structure = {
            "group1": {
                "regression": (str(pathlib.Path(tmp_path / "datasets/regression.csv")), None)
            },
            "group2": {
                "regression": (str(pathlib.Path(tmp_path / "datasets/regression.csv")), None)
            }
        }
        
        assert output_structure == expected_output_structure

    def test_get_output_structure_with_sql(self, mock_brisk_project, tmp_path):
        """Test the _get_output_structure method of ConfigurationManager what SQL table name is used."""        
        group1 = ExperimentGroup(
            name="group1",
            datasets=[("test_data.db", "regression")],
            algorithms=["linear"],
        )

        manager = ConfigurationManager([group1], {})
        output_structure = manager._get_output_structure()
        
        expected_output_structure = {
            "group1": {
                "test_data_regression": (str(pathlib.Path(tmp_path / "datasets/test_data.db")), "regression")
            }
        }
        assert output_structure == expected_output_structure

    def test_get_output_structure_with_multiple_datasets(self, mock_brisk_project, tmp_path):
        """Test the _get_output_structure method of ConfigurationManager with multiple datasets."""
        group1 = ExperimentGroup(
            name="group1",
            datasets=["regression.csv", "categorical.csv"],
            algorithms=["linear"]
        )
        group2 = ExperimentGroup(
            name="group2",
            datasets=[
                "regression.csv",
                ("test_data.db", "regression"),
                ("test_data.db", "categorical")
            ],
            algorithms=["ridge"]
        )
        manager = ConfigurationManager([group1, group2], {})
        output_structure = manager._get_output_structure()
        expected_output_structure = {
            "group1": {
                "regression": (str(pathlib.Path(tmp_path / "datasets/regression.csv")), None),
                "categorical": (str(pathlib.Path(tmp_path / "datasets/categorical.csv")), None)
            },
            "group2": {
                "regression": (str(pathlib.Path(tmp_path / "datasets/regression.csv")), None),
                "test_data_regression": (str(pathlib.Path(tmp_path / "datasets/test_data.db")), "regression"),
                "test_data_categorical": (str(pathlib.Path(tmp_path / "datasets/test_data.db")), "categorical")
            }
        }
        assert output_structure == expected_output_structure

    def test_create_description_map(self, mock_brisk_project):
        """Test the _create_description_map method of ConfigurationManager."""
        group1 = ExperimentGroup(
            name="group1",
            datasets=["regression.csv"],
            algorithms=["linear"],
            description="This is a test description"
        )
        group2 = ExperimentGroup(
            name="group2",
            datasets=["regression.csv"],
            algorithms=["ridge"],
            description="This is another test description that needs to be wrapped and stored properly."
        )
        group3 = ExperimentGroup(
            name="group3",
            datasets=["regression.csv"],
            algorithms=["elasticnet"]
        )
        manager = ConfigurationManager([group1, group2, group3], {})
        expected_group2_description = textwrap.dedent("""
        This is another test description that needs to be wrapped
        and stored properly.
        """).strip()

        assert manager.description_map == {
            "group1": "This is a test description",
            "group2": expected_group2_description
        }
