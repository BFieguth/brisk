import pytest
import sklearn.metrics as metrics
import numpy as np
import inspect

from brisk.evaluation.metric_wrapper import MetricWrapper

@pytest.fixture
def metric_wrapper():
    wrapper = MetricWrapper(
        name="accuracy",
        func=metrics.accuracy_score,
        display_name="Accuracy",
        abbr="ac",
        normalize=True
    )
    return wrapper

class TestMetricWrapper:
    def test_set_params(self, metric_wrapper):
        metric_wrapper.set_params(normalize=False)
        assert metric_wrapper.params["normalize"] is False
        assert "split_metadata" in metric_wrapper.params

    def test_ensure_split_metadata_param(self, metric_wrapper):
        func = metric_wrapper._ensure_split_metadata_param(
            metrics.accuracy_score
        )       
        assert func.__name__ == "accuracy_score"
        assert func.__qualname__ == "accuracy_score"
        assert func.__doc__ == metrics.accuracy_score.__doc__
        assert "split_metadata" in inspect.signature(func).parameters

    def test_function_is_callable(self, metric_wrapper):
        func_with_params = metric_wrapper.get_func_with_params()

        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 2])

        result = func_with_params(y_true, y_pred)
        expected_result = metrics.accuracy_score(y_true, y_pred)
        assert result == expected_result
