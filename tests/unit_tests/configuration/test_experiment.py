import pytest
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from brisk.configuration.experiment import Experiment
from brisk.utility.algorithm_wrapper import AlgorithmWrapper

@pytest.fixture
def linear_wrapper():
    """Create a LinearRegression AlgorithmWrapper."""
    return AlgorithmWrapper(
        name="linear",
        display_name="Linear Regression",
        algorithm_class=LinearRegression,
        default_params={'fit_intercept': True},
        hyperparam_grid={'fit_intercept': [True, False]}
    )

@pytest.fixture
def rf_wrapper():
    """Create a RandomForest AlgorithmWrapper."""
    return AlgorithmWrapper(
        name="rf",
        display_name="Random Forest",
        algorithm_class=RandomForestRegressor,
        default_params={'n_estimators': 100},
        hyperparam_grid={'n_estimators': [50, 100, 200]}
    )

@pytest.fixture
def ridge_wrapper():
    """Create a Ridge AlgorithmWrapper."""
    return AlgorithmWrapper(
        name="ridge",
        display_name="Ridge Regression",
        algorithm_class=Ridge,
        default_params={'alpha': 1.0},
        hyperparam_grid={'alpha': [0.1, 1.0, 10.0]}
    )

@pytest.fixture
def single_model(linear_wrapper):
    """Create a simple experiment with one model."""
    return Experiment(
        group_name="test_group",
        dataset_path=Path("data/test.csv"),
        algorithms={"model": linear_wrapper},
        workflow_args={"metrics": ["MAE", "MSE"]},
        table_name=None,
        categorical_features=None
    )

@pytest.fixture
def multiple_models(linear_wrapper, rf_wrapper):
    """Create an experiment with multiple models."""
    return Experiment(
        group_name="test_group",
        dataset_path=Path("data/test.csv"),
        algorithms={
            "model": linear_wrapper,
            "model2": rf_wrapper
        },
        workflow_args={},
        table_name=None,
        categorical_features=None
    )

@pytest.fixture
def long_name(linear_wrapper):
    """Create a simple experiment with one model."""
    return Experiment(
        group_name="a_very_long_group_name_indeed",
        dataset_path=Path("data/test.csv"),
        algorithms={"model": linear_wrapper},
        workflow_args={},
        table_name=None,
        categorical_features=None
    )


class TestExperiment:
    def test_valid_single_model(self, single_model, linear_wrapper):
        """Test creation with single model."""
        assert single_model.group_name == "test_group"
        assert single_model.dataset_path == Path("data/test.csv")
        assert len(single_model.algorithms) == 1
        assert isinstance(single_model.algorithms["model"], AlgorithmWrapper)
        assert single_model.algorithms["model"].algorithm_class == LinearRegression
        assert single_model.workflow_args == {"metrics": ["MAE", "MSE"]}
        assert isinstance(single_model.algorithm_kwargs["model"], LinearRegression)
        assert single_model.algorithm_names == ["linear"]
        workflow_attrs = single_model.workflow_attributes
        assert set(workflow_attrs.keys()) == {"metrics", "model"}
        assert workflow_attrs["metrics"] == ["MAE", "MSE"]
        assert isinstance(workflow_attrs["model"], LinearRegression)

    def test_valid_multiple_models(self, multiple_models):
        """Test creation with multiple models."""
        assert len(multiple_models.algorithms) == 2
        assert isinstance(multiple_models.algorithms["model"], AlgorithmWrapper)
        assert isinstance(multiple_models.algorithms["model2"], AlgorithmWrapper)
        assert multiple_models.algorithms["model"].algorithm_class == LinearRegression
        assert multiple_models.algorithms["model2"].algorithm_class == RandomForestRegressor
        assert multiple_models.workflow_args == {}
        assert isinstance(multiple_models.algorithm_kwargs["model"], LinearRegression)
        assert isinstance(multiple_models.algorithm_kwargs["model2"], RandomForestRegressor)
        assert multiple_models.algorithm_names == ["linear", "rf"]
        workflow_attrs = multiple_models.workflow_attributes
        assert set(workflow_attrs.keys()) == {"model", "model2"}
        assert isinstance(workflow_attrs["model"], LinearRegression)
        assert isinstance(workflow_attrs["model2"], RandomForestRegressor)

    def test_invalid_model_keys(self, linear_wrapper, ridge_wrapper):
        """Test validation of model naming convention."""
        with pytest.raises(ValueError, match="Single model must use key"):
            Experiment(
                group_name="test",
                dataset_path="test.csv",
                algorithms={"wrong_key": linear_wrapper},
                workflow_args={},
                table_name=None,
                categorical_features=None
            )

        with pytest.raises(ValueError, match="Multiple models must use keys"):
            Experiment(
                group_name="test",
                dataset_path="test.csv",
                algorithms={
                    "model1": linear_wrapper,
                    "wrong_key": ridge_wrapper
                },
                workflow_args={},
                table_name=None,
                categorical_features=None
            )

    def test_name_format(self, long_name):
        """Test name property format."""
        assert long_name.name == "a_very_long_group_name_indeed_linear"

    def test_invalid_group_name(self, linear_wrapper):
        """Test validation of group name."""
        with pytest.raises(ValueError, match="Group name must be a string"):
            Experiment(
                group_name=123,
                dataset_path="test.csv",
                algorithms={"model": linear_wrapper},
                workflow_args={},
                table_name=None,
                categorical_features=None
            )

    def test_invalid_algorithms(self, linear_wrapper):
        """Test validation of algorithms."""
        with pytest.raises(ValueError, match="Algorithms must be a dictionary"):
            Experiment(
                group_name="test",
                dataset_path="test.csv",
                algorithms=[linear_wrapper],
                workflow_args={},
                table_name=None,
                categorical_features=None
            )
        
    def test_missing_algorithms(self, linear_wrapper):
        """Test validation of algorithms."""
        with pytest.raises(ValueError, match="At least one algorithm must be provided"):
            Experiment(
                group_name="test",
                dataset_path="test.csv",
                algorithms={},
                workflow_args={},
                table_name=None,
                categorical_features=None
            )
