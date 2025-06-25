import pytest
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
import importlib

from brisk.configuration.experiment_factory import ExperimentFactory
from brisk.configuration.experiment_group import ExperimentGroup
from brisk.configuration.algorithm_wrapper import AlgorithmWrapper

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


@pytest.fixture
def factory_categorical(mock_reg_algorithms_py, tmp_path):
    """Create ExperimentFactory instance."""
    algorithms_path = tmp_path / 'algorithms.py'
    spec = importlib.util.spec_from_file_location(
        "algorithms", str(algorithms_path)
        )
    algorithms_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(algorithms_module)
    algorithm_config = algorithms_module.ALGORITHM_CONFIG
    return ExperimentFactory(algorithm_config, {
        "categorical.csv": ["category"]
    })


class TestExperimentFactory:
    def test_initalization(self, mock_reg_algorithms_py, tmp_path):
        algorithms_path = tmp_path / 'algorithms.py'
        spec = importlib.util.spec_from_file_location(
            "algorithms", str(algorithms_path)
            )
        algorithms_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(algorithms_module)
        algorithm_config = algorithms_module.ALGORITHM_CONFIG
        factory = ExperimentFactory(algorithm_config, {})
        assert factory.algorithm_config == algorithm_config
        assert factory.categorical_features == {}

    def test_init_error(self):
        """Test error is thrown if algorithm_config is not an AlgorithmCollection"""
        with pytest.raises(TypeError, match="algorithm_config must be an AlgorithmCollection"):
            ExperimentFactory(None, {})

        with pytest.raises(TypeError, match="algorithm_config must be an AlgorithmCollection"):
            ExperimentFactory([], {})

        with pytest.raises(TypeError, match="algorithm_config must be an AlgorithmCollection"):
            ExperimentFactory({}, {})

        with pytest.raises(TypeError, match="algorithm_config must be an AlgorithmCollection"):
            algorithm_list = [
                AlgorithmWrapper("linear", "Linear Regression", LinearRegression),
                AlgorithmWrapper("ridge", "Ridge Regression", Ridge)
            ]
            ExperimentFactory(algorithm_list, {})
        
    def test_single_algorithm(self, factory, mock_regression_project, tmp_path):
        """Test creation of experiment with single algorithm."""
        group = ExperimentGroup(
            name="test",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 1
        exp = experiments[0]
        assert exp.group_name == "test"
        assert len(exp.algorithms) == 1
        assert "model" in exp.algorithms
        assert isinstance(exp.algorithms["model"], AlgorithmWrapper)
        assert isinstance(exp.algorithms["model"].algorithm_class, type(LinearRegression))
        assert exp.dataset_path == tmp_path / "datasets" / "regression.csv"
        assert exp.workflow_args == None
        assert exp.table_name == None
        assert exp.categorical_features == None

    def test_multiple_separate_algorithms(self, factory, mock_regression_project, tmp_path):
        """Test creation of separate experiments for multiple algorithms."""
        group = ExperimentGroup(
            name="test",
            datasets=["regression.csv"],
            algorithms=["linear", "ridge"]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 2

        # Linear Experiment
        linear_exp = experiments[0]
        assert linear_exp.group_name == "test"
        assert len(linear_exp.algorithms) == 1
        assert "model" in linear_exp.algorithms
        assert isinstance(linear_exp.algorithms["model"], AlgorithmWrapper)
        assert isinstance(linear_exp.algorithms["model"].algorithm_class, type(LinearRegression))
        assert linear_exp.dataset_path == tmp_path / "datasets" / "regression.csv"
        assert linear_exp.workflow_args == None
        assert linear_exp.table_name == None
        assert linear_exp.categorical_features == None

        # Ridge Experiment
        ridge_exp = experiments[1]
        assert ridge_exp.group_name == "test"
        assert len(ridge_exp.algorithms) == 1
        assert "model" in ridge_exp.algorithms
        assert isinstance(ridge_exp.algorithms["model"], AlgorithmWrapper)
        assert isinstance(ridge_exp.algorithms["model"].algorithm_class, type(Ridge))
        assert ridge_exp.dataset_path == tmp_path / "datasets" / "regression.csv"
        assert ridge_exp.workflow_args == None
        assert ridge_exp.table_name == None
        assert ridge_exp.categorical_features == None

    def test_combined_algorithms(self, factory, mock_regression_project, tmp_path):
        """Test creation of single experiment with multiple algorithms."""
        group = ExperimentGroup(
            name="test",
            datasets=["regression.csv"],
            algorithms=[["linear", "ridge"]]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 1
        exp = experiments[0]
        assert len(exp.algorithms) == 2
        assert "model" in exp.algorithms
        assert "model2" in exp.algorithms
        assert isinstance(exp.algorithms["model"], AlgorithmWrapper)
        assert isinstance(exp.algorithms["model2"], AlgorithmWrapper)
        assert isinstance(exp.algorithms["model"].algorithm_class, type(LinearRegression))
        assert isinstance(exp.algorithms["model2"].algorithm_class, type(Ridge))
        assert exp.dataset_path == tmp_path / "datasets" / "regression.csv"
        assert exp.workflow_args == None
        assert exp.table_name == None
        assert exp.categorical_features == None

    def test_multiple_datasets(self, factory, mock_regression_project, tmp_path):
        """Test creation of experiments for multiple datasets."""      
        group = ExperimentGroup(
            name="test",
            datasets=["regression.csv", "group.csv"],
            algorithms=["linear"]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 2
        # Linear Experiment
        exp = experiments[0]
        assert exp.group_name == "test"
        assert len(exp.algorithms) == 1
        assert "model" in exp.algorithms
        assert isinstance(exp.algorithms["model"], AlgorithmWrapper)
        assert isinstance(exp.algorithms["model"].algorithm_class, type(LinearRegression))
        assert exp.dataset_path == tmp_path / "datasets" / "regression.csv"
        assert exp.workflow_args == None
        assert exp.table_name == None
        assert exp.categorical_features == None

    def test_algorithm_config(self, factory, mock_regression_project):
        """Test application of algorithm configuration."""
        group = ExperimentGroup(
            name="test",
            datasets=["regression.csv"],
            algorithms=["elasticnet"],
            algorithm_config={
                "elasticnet": {
                    "alpha": [0.1, 0.2, 0.3]
                }
            }
        )
        
        experiments = factory.create_experiments(group)
        exp = experiments[0]
        
        # Check hyperparameter grid was updated
        assert "alpha" in exp.algorithms["model"].hyperparam_grid
        assert exp.algorithms["model"].hyperparam_grid["alpha"] == [0.1, 0.2, 0.3]
        
        # Check default params weren't modified
        assert exp.algorithms["model"].default_params["alpha"] == 0.1
        assert exp.algorithms["model"].default_params["max_iter"] == 10000

    def test_invalid_algorithm(self, factory, mock_regression_project):
        """Test handling of invalid algorithm name."""
        group = ExperimentGroup(
            name="test",
            datasets=["regression.csv"],
            algorithms=["invalid_algo"]
        )
        
        with pytest.raises(KeyError, match="No algorithm found with name: "):
            factory.create_experiments(group)

    def test_mixed_algorithm_groups(self, factory, mock_regression_project, tmp_path):
        """Test handling of mixed single and grouped algorithms."""
        group = ExperimentGroup(
            name="test",
            datasets=["regression.csv"],
            algorithms=["linear", ["ridge", "elasticnet"]]
        )
        
        experiments = factory.create_experiments(group)
        assert len(experiments) == 2
        
        # Check single algorithm experiment
        single = next(exp for exp in experiments if len(exp.algorithms) == 1)
        assert "model" in single.algorithms
        assert isinstance(single.algorithms["model"], AlgorithmWrapper)
        assert isinstance(single.algorithms["model"].algorithm_class, type(LinearRegression))
        assert single.dataset_path == tmp_path / "datasets" / "regression.csv"
        assert single.workflow_args == None
        assert single.table_name == None
        assert single.categorical_features == None

        # Check grouped algorithm experiment
        grouped = next(exp for exp in experiments if len(exp.algorithms) == 2)
        assert "model" in grouped.algorithms
        assert "model2" in grouped.algorithms
        assert isinstance(grouped.algorithms["model"], AlgorithmWrapper)
        assert isinstance(grouped.algorithms["model2"], AlgorithmWrapper)
        assert isinstance(grouped.algorithms["model"].algorithm_class, type(Ridge))
        assert isinstance(grouped.algorithms["model2"].algorithm_class, type(ElasticNet))
        assert grouped.dataset_path == tmp_path / "datasets" / "regression.csv"
        assert grouped.workflow_args == None
        assert grouped.table_name == None
        assert grouped.categorical_features == None

    def test_create_experiment_categorical_features(self, factory_categorical, mock_regression_project, tmp_path):
        group = ExperimentGroup(
            name="test",
            datasets=["categorical.csv"],
            algorithms=["linear"]
        )
        experiments = factory_categorical.create_experiments(group)
        assert len(experiments) == 1
        exp = experiments[0]
        assert exp.group_name == "test"
        assert len(exp.algorithms) == 1
        assert "model" in exp.algorithms
        assert isinstance(exp.algorithms["model"], AlgorithmWrapper)
        assert isinstance(exp.algorithms["model"].algorithm_class, type(LinearRegression))
        assert exp.dataset_path == tmp_path / "datasets" / "categorical.csv"
        assert exp.workflow_args == None
        assert exp.table_name == None
        assert exp.categorical_features == ["category"]

    def test_create_experiment_sql(self, factory, mock_regression_project, tmp_path):
        group = ExperimentGroup(
            name="test",
            datasets=[("test_data.db", "regression")],
            algorithms=["linear"]
        )
        experiments = factory.create_experiments(group)
        assert len(experiments) == 1
        exp = experiments[0]
        assert exp.group_name == "test"
        assert len(exp.algorithms) == 1
        assert "model" in exp.algorithms
        assert isinstance(exp.algorithms["model"], AlgorithmWrapper)
        assert isinstance(exp.algorithms["model"].algorithm_class, type(LinearRegression))
        assert exp.dataset_path == tmp_path / "datasets" / "test_data.db"
        assert exp.table_name == "regression"
        assert exp.categorical_features == None

    def test_get_algorithm_wrapper(self, factory):
        """Test getting an algorithm wrapper."""
        wrapper = factory._get_algorithm_wrapper("linear")
        assert wrapper.name == "linear"
        assert wrapper.display_name == "Linear Regression"
        assert wrapper.algorithm_class == LinearRegression
        assert wrapper.default_params == {}
        assert wrapper.hyperparam_grid == {}

        wrapper = factory._get_algorithm_wrapper("ridge")
        assert wrapper.name == "ridge"
        assert wrapper.display_name == "Ridge Regression"
        assert wrapper.algorithm_class == Ridge
        assert wrapper.default_params == {"max_iter": 10000}
        assert wrapper.hyperparam_grid == {"alpha": [0.1, 0.5, 1.0]}

        wrapper = factory._get_algorithm_wrapper("elasticnet")
        assert wrapper.name == "elasticnet"
        assert wrapper.display_name == "Elastic Net Regression"
        assert wrapper.algorithm_class == ElasticNet
        assert wrapper.default_params == {"alpha": 0.1, "max_iter": 10000}
        assert wrapper.hyperparam_grid == {"alpha": [0.1, 0.2, 0.5], "l1_ratio": [0.1, 0.5, 1.0]}

    def test_get_algorithm_wrapper_with_config(self, factory):
        """Test getting an algorithm wrapper with configuration."""
        # Override the default hyperparameters
        wrapper = factory._get_algorithm_wrapper("ridge", {"alpha": 0.5})
        assert wrapper.name == "ridge"
        assert wrapper.display_name == "Ridge Regression"
        assert wrapper.algorithm_class == Ridge
        assert wrapper.default_params == {"max_iter": 10000}
        assert wrapper.hyperparam_grid == {"alpha": 0.5}

        # Check config does not change hyperparameters that are not specified
        wrapper = factory._get_algorithm_wrapper(
            "elasticnet",
            {
                "alpha": [0.5, 0.6, 0.7, 0.8],
                "fit_intercept": [True, False]
            }
        )
        assert wrapper.name == "elasticnet"
        assert wrapper.display_name == "Elastic Net Regression"
        assert wrapper.algorithm_class == ElasticNet
        assert wrapper.default_params == {"alpha": 0.1, "max_iter": 10000}
        assert wrapper.hyperparam_grid == {
            "alpha": [0.5, 0.6, 0.7, 0.8],
            "l1_ratio": [0.1, 0.5, 1.0],
            "fit_intercept": [True, False]
        }

    def test_normalize_algorithms(self, factory):
        """Test normalizing algorithms."""
        normalized = factory._normalize_algorithms(["linear", "ridge"])
        assert normalized == [["linear"], ["ridge"]]
        
        normalized = factory._normalize_algorithms([["linear", "ridge"]])
        assert normalized == [["linear", "ridge"]]

        normalized = factory._normalize_algorithms(["linear", ["ridge", "elasticnet"]])
        assert normalized == [["linear"], ["ridge", "elasticnet"]]

        normalized = factory._normalize_algorithms(["linear", "ridge", "elasticnet", ["lasso", "ridge"], ["ridge"]])
        assert normalized == [["linear"], ["ridge"], ["elasticnet"], ["lasso", "ridge"], ["ridge"]]

    def test_normalize_algorithms_throws_error(self, factory):
        """Test error is thrown if algorithms is not a list."""
        with pytest.raises(TypeError, match="algorithms must be a list, got"):
            factory._normalize_algorithms("linear, ridge, elasticnet")

        with pytest.raises(TypeError, match="algorithms must be a list, got"):
            factory._normalize_algorithms({"linear", "ridge", "elasticnet"})

        with pytest.raises(TypeError, match="algorithms must contain strings or lists of strings, got"):
            factory._normalize_algorithms(["linear", 1, "elasticnet"])

        with pytest.raises(TypeError, match="algorithms must contain strings or lists of strings, got"):
            factory._normalize_algorithms(["linear", ["ridge", 1], "elasticnet"])

        with pytest.raises(TypeError, match="algorithms must contain strings or lists of strings, got"):
            factory._normalize_algorithms(["linear", ["ridge", "elasticnet"], {"linear", "ridge"}])
