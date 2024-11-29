import numpy as np

from brisk.defaults.regression_metrics import concordance_correlation_coefficient

def test_concordance_correlation_coefficient():
    """
    Test the concordance correlation coefficient calculation.
    """
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])        
    ccc = concordance_correlation_coefficient(
        y_true, y_pred
        )
    assert np.isclose(ccc, 0.976, atol=0.01)
    