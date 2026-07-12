"""Integration tests for AlgorithmWrapper."""
import json

import pytest
from sklearn import linear_model

from brisk.adapters.sklearn import model_adapter as algorithm_wrapper


@pytest.mark.integration
class TestAlgorithmWrapperIntegration():
    """Integration tests for the AlgorithmWrapper class."""

    def test_export_config_serializable(self):
        wrapper = algorithm_wrapper.SklearnAlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge
        )
        output = wrapper.export_config()

        output_json = json.dumps(output, indent=2)
        deserialized = json.loads(output_json)

        assert deserialized == output

    def test_export_config_algorithm_module(self):
        wrapper = algorithm_wrapper.SklearnAlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge
        )
        output = wrapper.export_config()

        assert output["algorithm_class_module"] == "sklearn.linear_model._ridge"
        assert output["algorithm_class_name"] == "Ridge"
