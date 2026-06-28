"""Unit tests for the default sklearn algorithm and metric registrations.

Verifies that ``brisk.adapters.sklearn.defaults`` builds valid collections on
top of the sklearn adapter wrappers, mirroring the guarantees the original
``brisk.defaults`` modules provided (instantiable algorithms, scorer-capable
metrics, unique names) and that the entries conform to the algorithm/metric
ports.
"""

import numpy as np
import pytest

from brisk.adapters.sklearn import defaults
from brisk.adapters.sklearn.metric_adapter import SklearnMetricWrapper
from brisk.adapters.sklearn.model_adapter import SklearnAlgorithmWrapper
from brisk.ports import algorithm as algorithm_port
from brisk.ports import metric as metric_port

# pylint: disable=C0103

ALGORITHM_COLLECTIONS = [
    defaults.CLASSIFICATION_ALGORITHMS,
    defaults.REGRESSION_ALGORITHMS,
]
METRIC_COLLECTIONS = [
    defaults.CLASSIFICATION_METRICS,
    defaults.REGRESSION_METRICS,
]
ALL_ALGORITHMS = [w for c in ALGORITHM_COLLECTIONS for w in c]
ALL_METRICS = [w for c in METRIC_COLLECTIONS for w in c]


@pytest.mark.unit
class TestDefaultAlgorithms:
    def test_expected_counts(self):
        assert len(defaults.CLASSIFICATION_ALGORITHMS) == 7
        assert len(defaults.REGRESSION_ALGORITHMS) == 10

    @pytest.mark.parametrize("collection", ALGORITHM_COLLECTIONS)
    def test_all_are_sklearn_algorithm_wrappers(self, collection):
        assert all(
            isinstance(w, SklearnAlgorithmWrapper) for w in collection
        )

    @pytest.mark.parametrize("collection", ALGORITHM_COLLECTIONS)
    def test_names_unique_within_collection(self, collection):
        names = [w.name for w in collection]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("wrapper", ALL_ALGORITHMS, ids=lambda w: w.name)
    def test_satisfies_algorithm_wrapper_port(self, wrapper):
        assert isinstance(wrapper, algorithm_port.AlgorithmWrapperPort)

    @pytest.mark.parametrize("wrapper", ALL_ALGORITHMS, ids=lambda w: w.name)
    def test_instantiate_produces_model_port(self, wrapper):
        model = wrapper.instantiate()
        assert isinstance(model, algorithm_port.ModelPort)
        assert model.wrapper_name == wrapper.name

    @pytest.mark.parametrize("wrapper", ALL_ALGORITHMS, ids=lambda w: w.name)
    def test_export_config_round_trip(self, wrapper):
        config = wrapper.export_config()
        assert config["name"] == wrapper.name
        assert "algorithm_class_name" in config
        assert "default_params" in config
        assert "hyperparam_grid" in config


@pytest.mark.unit
class TestDefaultMetrics:
    def test_expected_counts(self):
        assert len(defaults.CLASSIFICATION_METRICS) == 10
        assert len(defaults.REGRESSION_METRICS) == 14

    @pytest.mark.parametrize("collection", METRIC_COLLECTIONS)
    def test_all_are_sklearn_metric_wrappers(self, collection):
        assert all(isinstance(w, SklearnMetricWrapper) for w in collection)

    @pytest.mark.parametrize("collection", METRIC_COLLECTIONS)
    def test_names_unique_within_collection(self, collection):
        names = [w.name for w in collection]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("wrapper", ALL_METRICS, ids=lambda w: w.name)
    def test_get_func_with_params_is_metric_port(self, wrapper):
        func = wrapper.get_func_with_params()
        assert isinstance(func, metric_port.MetricPort)

    @pytest.mark.parametrize("wrapper", ALL_METRICS, ids=lambda w: w.name)
    def test_get_scorer_is_callable(self, wrapper):
        assert callable(wrapper.get_scorer())

    @pytest.mark.parametrize("wrapper", ALL_METRICS, ids=lambda w: w.name)
    def test_abbr_defaults_to_name(self, wrapper):
        assert wrapper.abbr is not None


@pytest.mark.unit
class TestCustomMetricFunctions:
    def test_ccc_perfect_agreement(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert defaults.concordance_correlation_coefficient(y, y) == \
            pytest.approx(1.0, abs=1e-6)

    def test_ccc_returns_float(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.7])
        result = defaults.concordance_correlation_coefficient(y_true, y_pred)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_adjusted_r2_perfect_fit(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = y_true.copy()
        result = defaults.adjusted_r2_score(
            y_true, y_pred, split_metadata={"num_features": 2}
        )
        assert isinstance(result, float)
        assert result == pytest.approx(1.0, abs=1e-6)
