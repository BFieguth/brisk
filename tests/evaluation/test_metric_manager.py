from unittest.mock import MagicMock

import pytest

from brisk.evaluation.MetricManager import MetricManager
from brisk.utility.MetricWrapper import MetricWrapper
from brisk.defaults.RegressionMetrics import REGRESSION_METRICS

class TestMetricManager:
    """Test class for MetricManager."""

    @pytest.fixture
    def metric_manager(self):
        """Fixture to initialize MetricManager."""
        return MetricManager(*REGRESSION_METRICS)

    def test_initialization_with_regression(self, metric_manager):
        """
        Test initialization of MetricManager with regression metrics included.
        """
        assert "mean_absolute_error" in metric_manager._metrics_by_name
        assert "MSE" in metric_manager._abbreviations_to_name 
        assert "CCC" in metric_manager._abbreviations_to_name 

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
        assert "TMU" in manager._abbreviations_to_name
        assert manager._abbreviations_to_name["TMU"] == "test_metric"

    def test_list_metrics(self, metric_manager):
        """Test that the list_metrics method returns a list of metric names."""
        metric_names = metric_manager.list_metrics()
        assert isinstance(metric_names, list)
        assert all(isinstance(name, str) for name in metric_names)
        assert len(metric_names) == len(REGRESSION_METRICS)
        
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
        assert all(name in metric_names for name in regression_metric_names)
        