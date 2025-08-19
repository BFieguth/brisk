import inspect

import pytest

from brisk.evaluation.metric_manager import MetricManager
from brisk.evaluation.metric_wrapper import MetricWrapper
from brisk.defaults.regression_metrics import REGRESSION_METRICS
from brisk.defaults.classification_metrics import CLASSIFICATION_METRICS

@pytest.fixture
def metric_manager():
    """Fixture to initialize MetricManager."""
    return MetricManager(*REGRESSION_METRICS, *CLASSIFICATION_METRICS)


class TestMetricManager:
    """Test class for MetricManager."""
    def test_initialization_with_regression(self, metric_manager):
        """
        Test initialization of MetricManager with regression metrics included.
        """
        expected_metrics = [
            (wrapper.name, wrapper.abbr, wrapper.display_name) for wrapper in [
                *REGRESSION_METRICS, *CLASSIFICATION_METRICS
            ]
        ]
        for name, abbr, display_name in expected_metrics:
            assert name in metric_manager._metrics_by_name
            assert abbr in metric_manager._abbreviations_to_name 
            assert display_name in metric_manager._display_name_to_name

    def test_add_metric_removes_old_abbreviation(self):
        """Test that updating a metric properly removes its old abbreviation."""
        manager = MetricManager()
        
        # Add initial metric with abbreviation
        initial_metric = MetricWrapper(
            name="test_metric",
            display_name="Test Metric",
            abbr="TM",
            func=lambda x, y: 0,
            greater_is_better=True
        )
        manager._add_metric(initial_metric)
        
        # Add new metric with same name but different abbreviation
        updated_metric = MetricWrapper(
            name="test_metric",
            display_name="Test Metric Updated",
            abbr="TMU",
            func=lambda x, y: 0,
            greater_is_better=True
        )
        manager._add_metric(updated_metric)
        
        assert "TM" not in manager._abbreviations_to_name
        assert "Test Metric" not in manager._display_name_to_name
        assert "TMU" in manager._abbreviations_to_name
        assert "Test Metric Updated" in manager._display_name_to_name
        assert manager._abbreviations_to_name["TMU"] == "test_metric"
        assert manager._display_name_to_name["Test Metric Updated"] == "test_metric"

    def test_resolve_identifier_by_name(self, metric_manager):
        """Test that the resolve_identifier method returns the correct name."""
        name = metric_manager._resolve_identifier("mean_squared_error")
        assert name == "mean_squared_error"

        name = metric_manager._resolve_identifier("MSE")
        assert name == "mean_squared_error"

        with pytest.raises(ValueError, match="Metric 'mse' not found"):
            metric_manager._resolve_identifier("mse")

        name = metric_manager._resolve_identifier("Mean Squared Error")
        assert name == "mean_squared_error"

    def test_get_metric_by_name(self, metric_manager):
        """
        Test get_metric method by full name.
        """
        scorer = metric_manager.get_metric("mean_squared_error")
        assert scorer is not None
        assert callable(scorer)

    def test_get_metric_by_abbreviation(self, metric_manager):
        """
        Test get_metric method by abbreviation.
        """
        scorer = metric_manager.get_metric("MSE")
        assert scorer is not None
        assert callable(scorer)

    def test_get_metric_by_display_name(self, metric_manager):
        scorer = metric_manager.get_metric("Mean Squared Error")
        assert scorer is not None
        assert callable(scorer)

    def test_get_metric_invalid(self, metric_manager):
        """
        Test get_metric method with an invalid name or abbreviation, 
        ensuring it raises a ValueError.
        """
        with pytest.raises(ValueError, match="Metric 'invalid_scorer' not found"):
            metric_manager.get_metric("invalid_scorer")

    def test_get_scorer_by_name(self, metric_manager):
        """
        Test get_scorer method by full metric name.
        """
        scorer = metric_manager.get_scorer("mean_squared_error")
        assert scorer is not None
        assert callable(scorer)

    def test_get_scorer_by_abbreviation(self, metric_manager):
        """
        Test get_scorer method by abbreviation.
        """
        scorer = metric_manager.get_scorer("MSE")
        assert scorer is not None
        assert callable(scorer)

    def test_get_scorer_by_display_name(self, metric_manager):
        scorer = metric_manager.get_metric("Mean Squared Error")
        assert scorer is not None
        assert callable(scorer)

    def test_get_scorer_invalid(self, metric_manager):
        """
        Test get_scorer method with an invalid name or abbreviation, 
        ensuring it raises a ValueError.
        """
        with pytest.raises(
            ValueError, match="Metric 'invalid_scorer' not found"
            ):
            metric_manager.get_scorer("invalid_scorer")

    def test_get_name_by_full_name(self, metric_manager):
        """
        Test get_name method to retrieve display name by full metric name.
        """
        display_name = metric_manager.get_name("mean_absolute_error")
        assert display_name == "Mean Absolute Error"

    def test_get_name_by_abbreviation(self, metric_manager):
        """
        Test get_name method to retrieve display name by abbreviation.
        """
        display_name = metric_manager.get_name("MSE")
        assert display_name == "Mean Squared Error"

    def test_get_name_invalid(self, metric_manager):
        """
        Test get_name method with an invalid name or abbreviation, 
        ensuring it raises a ValueError.
        """
        with pytest.raises(
            ValueError, match="Metric 'invalid_name' not found"
            ):
            metric_manager.get_name("invalid_name")

    def test_list_metrics(self, metric_manager):
        """Test that the list_metrics method returns a list of metric names."""
        metric_names = metric_manager.list_metrics()
        assert isinstance(metric_names, list)
        assert all(isinstance(name, str) for name in metric_names)
        assert len(metric_names) == len(REGRESSION_METRICS) + len(CLASSIFICATION_METRICS)
        
        regression_metric_names = [
            "explained_variance_score",
            "max_error",
            "mean_absolute_error",
            "mean_absolute_percentage_error",
            "mean_pinball_loss",
            "mean_squared_error",
            "mean_squared_log_error",
            "median_absolute_error",
            "r2_score",
            "root_mean_squared_error",
            "root_mean_squared_log_error",
            "concordance_correlation_coefficient",
            "neg_mean_absolute_error"
        ]
        classification_metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "balanced_accuracy",
            "top_k_accuracy",
            "log_loss",
            "roc_auc",
            "brier",
            "roc"
        ]
        assert all(name in metric_names for name in regression_metric_names)
        assert all(name in metric_names for name in classification_metric_names)

    def test_set_split_metadata(self, metric_manager):
        """Test that the split_metadata is set correctly for all metric 
        functions."""
        split_metadata_input = {
            "test_value": 10,
            "test_value_2": 5.3
        }
        metric_manager.set_split_metadata(split_metadata_input)
        for wrapper in metric_manager._metrics_by_name.values():
            func = wrapper.get_func_with_params()
            paramaters = inspect.signature(func).parameters
            assert paramaters["split_metadata"].default == split_metadata_input

    def test_is_higher_better(self, metric_manager):
        assert metric_manager.is_higher_better("MSE") is False
        assert metric_manager.is_higher_better("Mean Absolute Error") is False
        assert metric_manager.is_higher_better("CCC") is True
        assert metric_manager.is_higher_better("r2_score") is True
