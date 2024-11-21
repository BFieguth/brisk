import pytest
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
import numpy as np

from brisk.configuration.ExperimentFactory import ExperimentFactory
from brisk.configuration.ExperimentGroup import ExperimentGroup
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper

@pytest.fixture
def mock_algorithm_config():
    """Create mock algorithm configurations."""
    return [
        AlgorithmWrapper(
            name="linear",
            display_name="Linear Regression",
            algorithm_class=LinearRegression,
        ),
        AlgorithmWrapper(
            name="ridge",
            display_name="Ridge Regression",
            algorithm_class=Ridge,
            default_params={"alpha": 1.0},
            hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
        ),
        AlgorithmWrapper(
            name="elasticnet",
            display_name="Elastic Net",
            algorithm_class=ElasticNet,
            default_params={"alpha": 1.0, "l1_ratio": 0.5},
            hyperparam_grid={"alpha": np.logspace(-3, 0, 100), "l1_ratio": np.logspace(-3, 0, 100)}
        )
    ]


@pytest.fixture
def factory(mock_algorithm_config):
    """Create ExperimentFactory instance."""
    return ExperimentFactory(mock_algorithm_config)


class TestExperimentFactory:
    def test_single_algorithm(self, factory, mock_project_root, monkeypatch):
        """Test creation of experiment with single algorithm."""
        monkeypatch.chdir(mock_project_root)
        
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 1
        
        exp = experiments[0]
        assert exp.group_name == "test_data"
        assert len(exp.algorithms) == 1
        assert "model" in exp.algorithms
        assert isinstance(exp.algorithms["model"], LinearRegression)

    def test_multiple_separate_algorithms(self, factory, mock_project_root, monkeypatch):
        """Test creation of separate experiments for multiple algorithms."""
        monkeypatch.chdir(mock_project_root)
        
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["linear", "ridge"]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 2
        
        # Check each experiment has one model
        for exp in experiments:
            assert len(exp.algorithms) == 1
            assert "model" in exp.algorithms

    def test_combined_algorithms(self, factory, mock_project_root, monkeypatch):
        """Test creation of single experiment with multiple algorithms."""
        monkeypatch.chdir(mock_project_root)
        
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=[["linear", "ridge"]]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 1
        
        exp = experiments[0]
        assert len(exp.algorithms) == 2
        assert "model1" in exp.algorithms
        assert "model2" in exp.algorithms

    def test_multiple_datasets(self, factory, mock_project_root, monkeypatch):
        """Test creation of experiments for multiple datasets."""
        monkeypatch.chdir(mock_project_root)
        
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv", "data_OLD.csv"],
            algorithms=["linear"]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 2
        
        names = {exp.group_name for exp in experiments}
        assert names == {"test_data", "test_data_OLD"}

    def test_algorithm_config(self, factory, mock_project_root, monkeypatch):
        """Test application of algorithm configuration."""
        monkeypatch.chdir(mock_project_root)
        
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["elasticnet"],
            algorithm_config={
                "elasticnet": {
                    "alpha": np.logspace(-1, 1, 10)
                }
            }
        )
        
        experiments = factory.create_experiments(group)
        exp = experiments[0]
        
        # Check hyperparameter grid was updated
        assert "alpha" in exp.hyperparameters["model"]
        assert len(exp.hyperparameters["model"]["alpha"]) == 10
        
        # Check default params weren't modified
        assert exp.algorithms["model"].alpha == 1.0
        assert exp.algorithms["model"].l1_ratio == 0.5

    def test_invalid_algorithm(self, factory, mock_project_root, monkeypatch):
        """Test handling of invalid algorithm name."""
        monkeypatch.chdir(mock_project_root)
        
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["invalid_algo"]
        )
        
        with pytest.raises(KeyError, match="Algorithm 'invalid_algo' not found"):
            factory.create_experiments(group)

    def test_mixed_algorithm_groups(self, factory, mock_project_root, monkeypatch):
        """Test handling of mixed single and grouped algorithms."""
        monkeypatch.chdir(mock_project_root)
        
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["linear", ["ridge", "elasticnet"]]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 2
        
        # Check single algorithm experiment
        single = next(exp for exp in experiments if len(exp.algorithms) == 1)
        assert "model" in single.algorithms
        assert isinstance(single.algorithms["model"], LinearRegression)
        
        # Check grouped algorithm experiment
        grouped = next(exp for exp in experiments if len(exp.algorithms) == 2)
        assert "model1" in grouped.algorithms
        assert "model2" in grouped.algorithms
        