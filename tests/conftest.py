import pytest

from brisk.utility.utility import find_project_root

@pytest.fixture
def mock_briskconfig_file(tmp_path):
    """Create a .briskconfig file for testing."""
    briskconfig_path = tmp_path / '.briskconfig'
    with open(briskconfig_path, 'w') as f:
        f.write("project_name=brisk-testing")
    return briskconfig_path


@pytest.fixture
def mock_reg_algorithms_py(tmp_path):
    algorithm_path = tmp_path / 'algorithms.py'
    algorithm_py = """
import numpy as np
import sklearn.linear_model as linear
    
import brisk
    
ALGORITHM_CONFIG = [
    brisk.AlgorithmWrapper(
        name="linear",
        display_name="Linear Regression",
        algorithm_class=linear.LinearRegression
    ),
    brisk.AlgorithmWrapper(
        name="ridge",
        display_name="Ridge Regression",
        algorithm_class=linear.Ridge,
        default_params={"max_iter": 10000},
        hyperparam_grid={"alpha": [0.1, 0.5, 1.0]}
    ),
    brisk.AlgorithmWrapper(
        name="elasticnet",
        display_name="Elastic Net Regression",
        algorithm_class=linear.ElasticNet,
        default_params={"alpha": 0.1, "max_iter": 10000},
        hyperparam_grid={
            "alpha": [0.1, 0.2, 0.5],
            "l1_ratio": [0.1, 0.5, 1.0]
        }
    ),
]
"""
    algorithm_path.write_text(algorithm_py)
    return algorithm_path


@pytest.fixture
def mock_reg_data_py(tmp_path):
    data_path = tmp_path / 'data.py'
    data_py = """
from brisk.data.data_manager import DataManager

BASE_DATA_MANAGER = DataManager(
    test_size=0.2, 
    n_splits=5,
    split_method="shuffle",
    random_state=42
)
"""
    data_path.write_text(data_py)
    return data_path


@pytest.fixture
def mock_reg_metric_py(tmp_path):
    metric_path = tmp_path / 'metric.py'
    metric_py = """
import brisk

METRIC_CONFIG = brisk.MetricManager(
    *brisk.REGRESSION_METRICS
)
"""
    metric_path.write_text(metric_py)
    return metric_path


@pytest.fixture
def mock_reg_training_py(tmp_path):
    training_path = tmp_path / 'training.py'
    training_py = """
from brisk.training.TrainingManager import TrainingManager
from metrics import METRIC_CONFIG
from settings import create_configuration

config = create_configuration()

# Define the TrainingManager for experiments
manager = TrainingManager(
    metric_config=METRIC_CONFIG,
    config_manager=config
)             
"""
    training_path.write_text(training_py)
    return training_path


@pytest.fixture
def mock_reg_settings_py(tmp_path):
    settings_path = tmp_path / 'settings.py'
    settings_py = """
from brisk.configuration.Configuration import Configuration, ConfigurationManager

def create_configuration() -> ConfigurationManager:
    config = Configuration(
        default_algorithms = ["linear"],
    )

    return config.build()

WORKFLOW_CONFIG = {
    "kfold": 2,
    "scoring": "CCC",
    "num_repeats": 2,
}
"""
    settings_path.write_text(settings_py)
    return settings_path


@pytest.fixture
def mock_reg_datasets(tmp_path):
    """Create a datasets directory with sample files."""
    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()

    sample_data = {
        'data.csv': """x,y,target
1.0,2.0,0
2.0,3.0,1
3.0,4.0,0
4.0,5.0,1
5.0,6.0,0""",
        
        'data2.csv': """feature1,feature2,label
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
    
    for filename, content in sample_data.items():
        (datasets_dir / filename).write_text(content)

    return datasets_dir


@pytest.fixture
def mock_regression_project(
    mock_briskconfig_file,
    mock_reg_algorithms_py,
    mock_reg_data_py,
    mock_reg_metric_py,
    mock_reg_training_py,
    mock_reg_settings_py,
    mock_reg_datasets,
    tmp_path,
    monkeypatch
):
    """Create a standardized temporary project structure for testing."""    
    monkeypatch.chdir(tmp_path)
    return tmp_path

    
@pytest.fixture(autouse=True)
def reset_project_root_cache():
    """Clear the project root cache after each test"""
    yield
    find_project_root.cache_clear()
    