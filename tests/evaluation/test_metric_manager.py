from unittest.mock import MagicMock

import pytest

from brisk.evaluation.MetricManager import MetricManager
from brisk.config.RegressionMetrics import REGRESSION_METRICS

class TestMetricManager:
    """Test class for MetricManager."""

    @pytest.fixture
    def metric_manager(self):
        """Fixture to initialize MetricManager."""
        return MetricManager(REGRESSION_METRICS)

    def test_initialization_with_regression(self, metric_manager):
        """
        Test initialization of MetricManager with regression metrics included.
        """
        assert "mean_absolute_error" in metric_manager.scoring_metrics
        assert "MSE" in [metric_manager.scoring_metrics[key].get("abbr") 
                         for key in metric_manager.scoring_metrics]
        assert "CCC" in [metric_manager.scoring_metrics[key].get("abbr") 
                         for key in metric_manager.scoring_metrics]

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
        with pytest.raises(ValueError, match="Metric function 'invalid_scorer' not found"):
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
            ValueError, match="Scoring callable 'invalid_scorer' not found"
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
            ValueError, match="Scoring callable 'invalid_name' not found"
            ):
            metric_manager.get_name("invalid_name")
