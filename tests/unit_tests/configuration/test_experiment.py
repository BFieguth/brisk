import pytest
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from brisk.configuration.experiment import Experiment
from brisk.configuration.algorithm_wrapper import AlgorithmWrapper

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
        dataset_path=Path("datasets/regression.csv"),
        algorithms={"model": linear_wrapper},
        workflow_args={"metrics": ["MAE", "MSE"]},
        table_name=None,
        categorical_features=None,
        split_index=0
    )


@pytest.fixture
def multiple_models(linear_wrapper, rf_wrapper):
    """Create an experiment with multiple models."""
    return Experiment(
        group_name="test_group",
        dataset_path=Path("datasets/regression.csv"),
        algorithms={
            "model": linear_wrapper,
            "model2": rf_wrapper
        },
        workflow_args={},
        table_name=None,
        categorical_features=None,
        split_index=0
    )


@pytest.fixture
def sql_table(ridge_wrapper):
    """Create an experiment with a SQL table."""
    return Experiment(
        group_name="sql_table",
        dataset_path=Path("datasets/test_data.db"),
        algorithms={"model": ridge_wrapper},
        workflow_args={},
        table_name="categorical",
        categorical_features=["category"],
        split_index=0
    )


class TestExperiment:
    def test_valid_single_model(self, single_model, linear_wrapper):
        """Test creation with single model."""
        expected_model = linear_wrapper.instantiate()

        # group_name
        assert single_model.group_name == "test_group"
        # algorithms
        assert len(single_model.algorithms) == 1
        assert isinstance(single_model.algorithms["model"], AlgorithmWrapper)
        assert single_model.algorithms["model"].algorithm_class == LinearRegression
        # dataset_path
        assert single_model.dataset_path == Path("datasets/regression.csv")
        # workflow_args
        assert single_model.workflow_args == {"metrics": ["MAE", "MSE"]}
        # table_name
        assert single_model.table_name is None
        # categorical_features
        assert single_model.categorical_features is None
        # name
        assert single_model.name == "test_group_linear"
        # dataset_name
        assert single_model.dataset_name == ("regression", None)
        # algorithm_kwargs
        assert type(single_model.algorithm_kwargs["model"]) == type(expected_model)
        # algorithm_names
        assert single_model.algorithm_names == ["linear"]
        # workflow_attributes
        workflow_attrs = single_model.workflow_attributes
        assert set(workflow_attrs.keys()) == {"metrics", "model"}
        assert workflow_attrs["metrics"] == ["MAE", "MSE"]
        assert isinstance(workflow_attrs["model"], LinearRegression)

    def test_valid_multiple_models(self, multiple_models, linear_wrapper, rf_wrapper):
        """Test creation with multiple models."""
        expected_model_linear = linear_wrapper.instantiate()
        expected_model_rf = rf_wrapper.instantiate()

        # group_name
        assert multiple_models.group_name == "test_group"
        # algorithms
        assert len(multiple_models.algorithms) == 2
        assert isinstance(multiple_models.algorithms["model"], AlgorithmWrapper)
        assert isinstance(multiple_models.algorithms["model2"], AlgorithmWrapper)
        assert multiple_models.algorithms["model"].algorithm_class == LinearRegression
        assert multiple_models.algorithms["model2"].algorithm_class == RandomForestRegressor
        # dataset_path
        assert multiple_models.dataset_path == Path("datasets/regression.csv")
        # workflow_args
        assert multiple_models.workflow_args == {}
        # table_name
        assert multiple_models.table_name is None
        # categorical_features
        assert multiple_models.categorical_features is None
        # name
        assert multiple_models.name == "test_group_linear_rf"
        # dataset_name
        assert multiple_models.dataset_name == ("regression", None)
        # algorithm_kwargs
        assert type(multiple_models.algorithm_kwargs["model"]) == type(expected_model_linear)
        assert type(multiple_models.algorithm_kwargs["model2"]) == type(expected_model_rf)
        # algorithm_names
        assert multiple_models.algorithm_names == ["linear", "rf"]
        # workflow_attributes
        workflow_attrs = multiple_models.workflow_attributes
        assert set(workflow_attrs.keys()) == {"model", "model2"}
        assert isinstance(workflow_attrs["model"], LinearRegression)
        assert isinstance(workflow_attrs["model2"], RandomForestRegressor)

    def test_sql_table(self, sql_table, ridge_wrapper):
        """Test creation with sql database."""
        expected_model = ridge_wrapper.instantiate()

        # group_name
        assert sql_table.group_name == "sql_table"
        # algorithms
        assert len(sql_table.algorithms) == 1
        assert isinstance(sql_table.algorithms["model"], AlgorithmWrapper)
        assert sql_table.algorithms["model"].algorithm_class == Ridge
        # dataset_path
        assert sql_table.dataset_path == Path("datasets/test_data.db")
        # workflow_args
        assert sql_table.workflow_args == {}
        # table_name
        assert sql_table.table_name == "categorical"
        # categorical_features
        assert sql_table.categorical_features == ["category"]
        # name
        assert sql_table.name == "sql_table_ridge"
        # dataset_name
        assert sql_table.dataset_name == ("test_data", "categorical")
        # algorithm_kwargs
        assert type(sql_table.algorithm_kwargs["model"]) == type(expected_model)
        # algorithm_names
        assert sql_table.algorithm_names == ["ridge"]
        # workflow_attributes
        workflow_attrs = sql_table.workflow_attributes
        assert set(workflow_attrs.keys()) == {"model"}
        assert isinstance(workflow_attrs["model"], Ridge)

    def test_invalid_model_keys(self, linear_wrapper, ridge_wrapper):
        """Test validation of model naming convention."""
        with pytest.raises(ValueError, match="Single model must use key"):
            Experiment(
                group_name="test",
                dataset_path="test.csv",
                algorithms={"wrong_key": linear_wrapper},
                workflow_args={},
                table_name=None,
                categorical_features=None,
                split_index=0
            )

        with pytest.raises(ValueError, match="Multiple models must use keys"):
            Experiment(
                group_name="test",
                dataset_path="test.csv",
                algorithms={
                    "model": linear_wrapper,
                    "wrong_key": ridge_wrapper
                },
                workflow_args={},
                table_name=None,
                categorical_features=None,
                split_index=0
            )

    def test_invalid_group_name(self, linear_wrapper):
        """Test validation of group name."""
        with pytest.raises(ValueError, match="Group name must be a string"):
            Experiment(
                group_name=123,
                dataset_path="test.csv",
                algorithms={"model": linear_wrapper},
                workflow_args={},
                table_name=None,
                categorical_features=None,
                split_index=0
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
                categorical_features=None,
                split_index=0
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
                categorical_features=None,
                split_index=0
            )
