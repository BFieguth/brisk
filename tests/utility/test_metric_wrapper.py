import pytest
import sklearn.metrics as metrics

from brisk.utility.metric_wrapper import MetricWrapper

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
