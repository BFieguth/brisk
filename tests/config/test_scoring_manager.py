from unittest import mock

import numpy as np
import pytest

from ml_toolkit.config.metric_manager import MetricManager

class TestMetricManager:
    """Test class for MetricManager."""

    @pytest.fixture
    def scoring_manager(self):
        """Fixture to initialize MetricManager."""
        return MetricManager()

    def test_initialization_with_regression(self, scoring_manager):
        """
        Test initialization of MetricManager with regression metrics included.
        """
        assert "mean_absolute_error" in scoring_manager.scoring_metrics
        assert "MSE" in [scoring_manager.scoring_metrics[key].get("abbr") 
                         for key in scoring_manager.scoring_metrics]
        assert "CCC" in [scoring_manager.scoring_metrics[key].get("abbr") 
                         for key in scoring_manager.scoring_metrics]

    def test_get_scorer_by_name(self, scoring_manager):
        """
        Test get_scorer method by full name.
        """
        scorer = scoring_manager.get_scorer("mean_squared_error")
        assert scorer is not None
        assert callable(scorer)

    def test_get_scorer_by_abbreviation(self, scoring_manager):
        """
        Test get_scorer method by abbreviation.
        """
        scorer = scoring_manager.get_scorer("MSE")
        assert scorer is not None
        assert callable(scorer)

    def test_get_scorer_invalid(self, scoring_manager):
        """
        Test get_scorer method with an invalid name or abbreviation, 
        ensuring it raises a ValueError.
        """
        with pytest.raises(ValueError, match="Scorer 'invalid_scorer' not found"):
            scoring_manager.get_scorer("invalid_scorer")

    def test_concordance_correlation_coefficient(self):
        """
        Test the concordance correlation coefficient calculation.
        """
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])        
        ccc = MetricManager._MetricManager__concordance_correlation_coefficient(
            y_true, y_pred
            )
        assert np.isclose(ccc, 0.976, atol=0.01)
