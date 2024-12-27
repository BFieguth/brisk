from unittest.mock import MagicMock

import pandas as pd
import pytest

from brisk.evaluation.evaluation_manager import EvaluationManager
from brisk.training.workflow import Workflow

class MockWorkflow(Workflow): # pragma: no cover

    def workflow(self): 
        """Mock workflow implementation for testing."""
        pass


class TestWorkflowClass:
    """Test class for the Workflow base class."""

    @pytest.fixture
    def setup_workflow(self):
        """Fixture to initialize a Workflow instance with mock data."""
        evaluator = MagicMock(spec=EvaluationManager)
        X_train = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        X_test = pd.DataFrame({"feature1": [5, 6], "feature2": [7, 8]})
        y_train = pd.Series([0, 1])
        y_test = pd.Series([1, 0])
        output_dir = "mock_output_dir"
        algorithm_names = ["mock_model1", "mock_model2"]
        feature_names = ["feature1", "feature2"]
        algorithm_kwargs = {"model1": MagicMock(), "model2": MagicMock()}

        return MockWorkflow(
            evaluator=evaluator, 
            X_train=X_train, 
            X_test=X_test, 
            y_train=y_train, 
            y_test=y_test, 
            output_dir=output_dir, 
            algorithm_names=algorithm_names, 
            feature_names=feature_names, 
            algorithm_kwargs=algorithm_kwargs
        )

    def test_unpack_attributes_model_kwargs(self, setup_workflow):
        """Test that model_kwargs are unpacked as attributes correctly."""
        workflow = setup_workflow
        assert hasattr(workflow, "model1")
        assert hasattr(workflow, "model2")
        assert isinstance(workflow.model1, MagicMock)
        assert isinstance(workflow.model2, MagicMock)

    def test_missing_attribute_error(self, setup_workflow):
        """Test that accessing a missing attribute raises AttributeError with 
        a detailed message."""
        workflow = setup_workflow
        with pytest.raises(AttributeError, match="not_found"):
            _ = workflow.not_found

    def test_getattr_error_message(self, setup_workflow):
        """Test that the AttributeError message lists available attributes."""
        workflow = setup_workflow
        with pytest.raises(AttributeError) as exc:
            _ = workflow.not_found
        assert "Available attributes are:" in str(exc.value)

    def test_abstract_method(self):
        """Test that Workflow raises TypeError if instantiated without implementing 'workflow'."""
        evaluator = MagicMock()
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_train = pd.Series()
        y_test = pd.Series()
        output_dir = "output"
        algorithm_names = []
        feature_names = []
        algorithm_kwargs = {}

        with pytest.raises(TypeError):
            Workflow(
                evaluator=evaluator, 
                X_train=X_train, 
                X_test=X_test, 
                y_train=y_train, 
                y_test=y_test, 
                output_dir=output_dir, 
                algorithm_names=algorithm_names, 
                feature_names=feature_names, 
                algorithm_kwargs=algorithm_kwargs
            )

    def test_not_implemented_error_direct(self):
        """Test that directly calling the base class 'workflow' method raises NotImplementedError."""
        
        class TempWorkflow(Workflow):
            def workflow(self):
                super().workflow()

        evaluator = MagicMock()
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_train = pd.Series()
        y_test = pd.Series()
        output_dir = "output"
        algorithm_names = []
        feature_names = []
        algorithm_kwargs = {}

        temp_workflow = TempWorkflow(
            evaluator, X_train, X_test, y_train, y_test, output_dir, 
            algorithm_names, feature_names, algorithm_kwargs
        )
        
        with pytest.raises(
            NotImplementedError, 
            match="Subclass must implement the workflow method."
            ):
            temp_workflow.workflow()

    def test_method_delegation(self):
        """Test that unknown attributes are properly delegated to evaluator."""
        class MockEvaluator:
            def test_method(self):
                return "test_result"
        
        mock_evaluator = MockEvaluator()        
        workflow = MockWorkflow(
            evaluator=mock_evaluator,
            X_train=pd.DataFrame(),
            X_test=pd.DataFrame(),
            y_train=pd.Series(),
            y_test=pd.Series(),
            output_dir="",
            algorithm_names=[],
            feature_names=[],
            algorithm_kwargs={}
        )
        
        # Test that the method is properly delegated
        assert workflow.test_method() == "test_result"
        
        # Test that non-existent attributes raise AttributeError
        with pytest.raises(AttributeError, match="'nonexistent_method' not found"):
            workflow.nonexistent_method()
