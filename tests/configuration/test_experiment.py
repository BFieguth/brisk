import pytest
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from brisk.configuration.Experiment import Experiment

@pytest.fixture
def single_model():
    """Create a simple experiment with one model."""
    return Experiment(
        group_name="test_group",
        dataset=Path("data/test.csv"),
        algorithms={"model": LinearRegression()}
    )


@pytest.fixture
def multiple_models():
    """Create an experiment with multiple models."""
    return Experiment(
        group_name="test_group",
        dataset=Path("data/test.csv"),
        algorithms={
            "model1": LinearRegression(),
            "model2": RandomForestRegressor()
        }
    )


class TestExperiment:
    def test_valid_single_model(self, single_model):
        """Test creation with single model."""
        assert single_model.group_name == "test_group"
        assert single_model.dataset == Path("data/test.csv")
        assert len(single_model.algorithms) == 1
        assert isinstance(single_model.algorithms["model"], LinearRegression)

    def test_valid_multiple_models(self, multiple_models):
        """Test creation with multiple models."""
        assert len(multiple_models.algorithms) == 2
        assert isinstance(multiple_models.algorithms["model1"], LinearRegression)
        assert isinstance(multiple_models.algorithms["model2"], RandomForestRegressor)

    def test_invalid_model_keys(self):
        """Test validation of model naming convention."""
        with pytest.raises(ValueError, match="Single model must use key"):
            Experiment(
                group_name="test",
                dataset="test.csv",
                algorithms={"wrong_key": LinearRegression()}
            )

        with pytest.raises(ValueError, match="Multiple models must use keys"):
            Experiment(
                group_name="test",
                dataset="test.csv",
                algorithms={
                    "model1": LinearRegression(),
                    "wrong_key": Ridge()
                }
            )

    def test_string_to_path_conversion(self):
        """Test automatic conversion of string to Path."""
        exp = Experiment(
            group_name="test",
            dataset="test.csv",
            algorithms={"model": LinearRegression()}
        )
        assert isinstance(exp.dataset, Path)

    def test_full_name_format(self, multiple_models):
        """Test full_name property format."""
        expected = "test_group_LinearRegression_RandomForestRegressor"
        assert multiple_models.full_name == expected

    def test_experiment_name_consistency(self, single_model):
        """Test experiment_name is consistent across instantiations."""
        exp2 = Experiment(
            group_name="test_group",
            dataset="data/test.csv",
            algorithms={"model": LinearRegression()}
        )
        assert single_model.experiment_name == exp2.experiment_name

    def test_experiment_name_uniqueness(self):
        """Test experiment_name is unique for different configurations."""
        exp1 = Experiment(
            group_name="test_group",
            dataset="test.csv",
            algorithms={"model": LinearRegression()}
        )
        exp2 = Experiment(
            group_name="test_group",
            dataset="test.csv",
            algorithms={"model": RandomForestRegressor()}
        )
        assert exp1.experiment_name != exp2.experiment_name

    def test_get_model_kwargs(self, multiple_models):
        """Test get_model_kwargs returns correct format."""
        kwargs = multiple_models.get_model_kwargs()
        assert list(kwargs.keys()) == ["model1", "model2"]
        assert isinstance(kwargs["model1"], LinearRegression)
        assert isinstance(kwargs["model2"], RandomForestRegressor)

    def test_invalid_inputs(self):
        """Test validation of invalid inputs."""
        with pytest.raises(ValueError, match="Group name must be a string"):
            Experiment(
                group_name=123,
                dataset="test.csv",
                algorithms={"model": LinearRegression()}
            )

        with pytest.raises(ValueError, match="At least one algorithm must be provided"):
            Experiment(
                group_name="test",
                dataset="test.csv",
                algorithms={}
            )

        with pytest.raises(ValueError, match="Algorithms must be a dictionary"):
            Experiment(
                group_name="test",
                dataset="test.csv",
                algorithms=[LinearRegression()]
            )
            