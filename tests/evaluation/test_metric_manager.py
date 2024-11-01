from unittest import mock

import numpy as np
import pytest

from brisk.evaluation.MetricManager import MetricManager
from brisk.config.RegressionMetrics import REGRESSION_METRICS

class TestMetricManager:
    """Test class for MetricManager."""

    @pytest.fixture
    def scoring_manager(self):
        """Fixture to initialize MetricManager."""
        return MetricManager(REGRESSION_METRICS)

    def test_initialization_with_regression(self, scoring_manager):
        """
        Test initialization of MetricManager with regression metrics included.
        """
        assert "mean_absolute_error" in scoring_manager.scoring_metrics
        assert "MSE" in [scoring_manager.scoring_metrics[key].get("abbr") 
                         for key in scoring_manager.scoring_metrics]
        assert "CCC" in [scoring_manager.scoring_metrics[key].get("abbr") 
                         for key in scoring_manager.scoring_metrics]

    def test_get_metric_by_name(self, scoring_manager):
        """
        Test get_metric method by full name.
        """
        scorer = scoring_manager.get_metric("mean_squared_error")
        assert scorer is not None
        assert callable(scorer)

    def test_get_metric_by_abbreviation(self, scoring_manager):
        """
        Test get_metric method by abbreviation.
        """
        scorer = scoring_manager.get_metric("MSE")
        assert scorer is not None
        assert callable(scorer)

    def test_get_metric_invalid(self, scoring_manager):
        """
        Test get_metric method with an invalid name or abbreviation, 
        ensuring it raises a ValueError.
        """
        with pytest.raises(ValueError, match="Metric function 'invalid_scorer' not found"):
            scoring_manager.get_metric("invalid_scorer")
