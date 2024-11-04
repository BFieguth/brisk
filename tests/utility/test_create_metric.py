import unittest

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn import metrics

from brisk.utility.CreateMetric import create_metric 

def custom_absolute_error(y_true, y_pred): # pragma: no cover
    """Custom metric to calculate the mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


class TestCreateMetric(unittest.TestCase):
    def test_create_metric_with_abbr(self):
        metric_info = create_metric(
            func=custom_absolute_error, name="Custom Absolute Error", abbr="CAE"
            )

        self.assertEqual(metric_info["func"], custom_absolute_error)
        self.assertIsInstance(metric_info["scorer"], metrics._scorer._Scorer)
        self.assertEqual(metric_info["abbr"], "CAE")
        self.assertEqual(metric_info["display_name"], "Custom Absolute Error")

    def test_create_metric_without_abbr(self):
        metric_info = create_metric(
            func=custom_absolute_error, name="Custom Absolute Error"
            )

        self.assertEqual(metric_info["func"], custom_absolute_error)
        self.assertIsInstance(metric_info["scorer"], metrics._scorer._Scorer)
        self.assertIsNone(metric_info["abbr"])
        self.assertEqual(metric_info["display_name"], "Custom Absolute Error")
