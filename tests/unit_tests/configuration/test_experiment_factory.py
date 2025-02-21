import pytest
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
import numpy as np
import importlib

from brisk.configuration.experiment_factory import ExperimentFactory
from brisk.configuration.experiment_group import ExperimentGroup
from brisk.utility.algorithm_wrapper import AlgorithmWrapper

@pytest.fixture
def factory(mock_reg_algorithms_py, tmp_path):
    """Create ExperimentFactory instance."""
    algorithms_path = tmp_path / 'algorithms.py'
    spec = importlib.util.spec_from_file_location(
        "algorithms", str(algorithms_path)
        )
    algorithms_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(algorithms_module)
    algorithm_config = algorithms_module.ALGORITHM_CONFIG
    return ExperimentFactory(algorithm_config, {})


class TestExperimentFactory:
    def test_single_algorithm(self, factory, mock_regression_project):
        """Test creation of experiment with single algorithm."""
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["linear"]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 1
        
        exp = experiments[0]
        assert exp.group_name == "test"
        assert len(exp.algorithms) == 1
        assert "model" in exp.algorithms
        assert isinstance(exp.algorithms["model"], AlgorithmWrapper)
        assert exp.algorithms["model"].algorithm_class == LinearRegression

    def test_multiple_separate_algorithms(self, factory, mock_regression_project):
        """Test creation of separate experiments for multiple algorithms."""
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

    def test_combined_algorithms(self, factory, mock_regression_project):
        """Test creation of single experiment with multiple algorithms."""
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=[["linear", "ridge"]]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 1
        
        exp = experiments[0]
        assert len(exp.algorithms) == 2
        assert "model" in exp.algorithms
        assert "model2" in exp.algorithms

    def test_multiple_datasets(self, factory, mock_regression_project):
        """Test creation of experiments for multiple datasets."""      
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv", "data2.csv"],
            algorithms=["linear"]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 2
        
        names = {exp.group_name for exp in experiments}
        assert names == {"test", "test"}

    def test_algorithm_config(self, factory, mock_regression_project):
        """Test application of algorithm configuration."""
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
        assert "alpha" in exp.algorithms["model"].hyperparam_grid
        assert len(exp.algorithms["model"].hyperparam_grid["alpha"]) == 10
        
        # Check default params weren't modified
        assert exp.algorithms["model"].default_params["alpha"] == 0.1

    def test_invalid_algorithm(self, factory, mock_regression_project):
        """Test handling of invalid algorithm name."""
        group = ExperimentGroup(
            name="test",
            datasets=["data.csv"],
            algorithms=["invalid_algo"]
        )
        
        with pytest.raises(KeyError, match="No algorithm found with name: "):
            factory.create_experiments(group)

    def test_mixed_algorithm_groups(self, factory, mock_regression_project):
        """Test handling of mixed single and grouped algorithms."""
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
        assert isinstance(single.algorithms["model"], AlgorithmWrapper)
        assert single.algorithms["model"].algorithm_class == LinearRegression
        
        # Check grouped algorithm experiment
        grouped = next(exp for exp in experiments if len(exp.algorithms) == 2)
        assert "model" in grouped.algorithms
        assert "model2" in grouped.algorithms
        assert isinstance(grouped.algorithms["model"], AlgorithmWrapper)
        assert isinstance(grouped.algorithms["model2"], AlgorithmWrapper)
        assert grouped.algorithms["model"].algorithm_class == Ridge
        assert grouped.algorithms["model2"].algorithm_class == ElasticNet
