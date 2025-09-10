import inspect

import pytest
import sklearn.metrics as metrics
import numpy as np

from brisk.evaluation.metric_wrapper import MetricWrapper

@pytest.fixture
def accuracy_wrapper():
    wrapper = MetricWrapper(
        name="accuracy",
        func=metrics.accuracy_score,
        display_name="Accuracy",
        abbr="ac",
        normalize=True,
        greater_is_better=True
    )
    return wrapper


@pytest.fixture
def f1_wrapper():
    wrapper = MetricWrapper(
        name="f1",
        func=metrics.f1_score,
        display_name="F1 Score",
        greater_is_better=True
    )
    return wrapper


class TestMetricWrapper:
    def test_init(self, accuracy_wrapper, f1_wrapper):
        # accuracy
        assert accuracy_wrapper.name == "accuracy"
        assert callable(accuracy_wrapper.func)
        assert callable(accuracy_wrapper._func_with_params)
        assert callable(accuracy_wrapper.scorer)
        assert accuracy_wrapper.display_name == "Accuracy"
        assert accuracy_wrapper.abbr == "ac"
        assert accuracy_wrapper.params["normalize"] is True
        assert accuracy_wrapper.params["split_metadata"] == {}

        # f1
        assert f1_wrapper.name == "f1"
        assert callable(f1_wrapper.func)
        assert callable(f1_wrapper._func_with_params)
        assert callable(f1_wrapper.scorer)
        assert f1_wrapper.display_name == "F1 Score"
        assert f1_wrapper.abbr == "f1"
        assert f1_wrapper.params["split_metadata"] == {}

    def test_set_params(self, accuracy_wrapper, f1_wrapper):
        accuracy_wrapper.set_params(normalize=False)
        assert accuracy_wrapper.params["normalize"] is False
        assert accuracy_wrapper.params["split_metadata"] == {}

        f1_wrapper.set_params(average="macro", zero_division=0)
        assert f1_wrapper.params["average"] == "macro"
        assert f1_wrapper.params["zero_division"] == 0
        assert f1_wrapper.params["split_metadata"] == {}


    def test_function_is_callable(self, accuracy_wrapper):
        func_with_params = accuracy_wrapper.get_func_with_params()

        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 2])

        result = func_with_params(y_true, y_pred)
        expected_result = metrics.accuracy_score(y_true, y_pred)
        assert result == expected_result

    def test_ensure_split_metadata_param(self, accuracy_wrapper):
        func = accuracy_wrapper._ensure_split_metadata_param(
            metrics.accuracy_score
        )       
        assert func.__name__ == "accuracy_score"
        assert func.__qualname__ == "accuracy_score"
        assert func.__doc__ == metrics.accuracy_score.__doc__
        assert "split_metadata" in inspect.signature(func).parameters
