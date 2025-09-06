from unittest.mock import MagicMock
import importlib
from unittest import mock

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from brisk.evaluation.evaluation_manager import EvaluationManager
from brisk.training.workflow import Workflow
from brisk.services import get_services

class WorkflowSubclass(Workflow): # pragma: no cover
    def workflow(self, X_train, X_test, y_train, y_test, output_dir, feature_names): 
        """Mock workflow implementation for testing."""
        self.evaluate_model(
            self.model, self.X_train, self.y_train, "MAE", "evaluate_model"
        )


class InvalidWorkflowSubclass(Workflow): # pragma: no cover
    def not_a_workflow(self):
        pass


@pytest.fixture
def mock_services(mock_brisk_project):
    return get_services()
    


@pytest.fixture
def workflow_factory(mock_brisk_project, tmp_path, mock_services):
    """Fixture to initialize a Workflow instance with mock data."""
    def _create_workflow(workflow_class=WorkflowSubclass, workflow_attributes=None):
        with (
            mock.patch("brisk.data.data_manager.get_services", return_value=mock_services),
            mock.patch("brisk.data.data_split_info.get_services", return_value=mock_services),
            mock.patch("brisk.evaluation.evaluation_manager.get_services", return_value=mock_services)
        ):
            # Load algorithm config    
            algorithm_file = tmp_path / "algorithms.py"
            algorithm_spec = importlib.util.spec_from_file_location(
                "algorithms", str(algorithm_file)
            )
            algorithm_module = importlib.util.module_from_spec(algorithm_spec)
            algorithm_spec.loader.exec_module(algorithm_module)
            algorithm_config = algorithm_module.ALGORITHM_CONFIG

            # Load metric config    
            metric_file = tmp_path / "metrics.py"
            metric_spec = importlib.util.spec_from_file_location(
                "metrics", str(metric_file)
            )
            metric_module = importlib.util.module_from_spec(metric_spec)
            metric_spec.loader.exec_module(metric_module)
            metric_config = metric_module.METRIC_CONFIG

            # Setup EvaluationManager
            evaluation_manager = EvaluationManager(metric_config)

            # Load data
            data_file = tmp_path / "data.py"
            data_spec = importlib.util.spec_from_file_location(
                "data", str(data_file)
            )
            data_module = importlib.util.module_from_spec(data_spec)
            data_spec.loader.exec_module(data_module)
            data_manager = data_module.BASE_DATA_MANAGER
            data_split = data_manager.split(
                data_path=tmp_path / "datasets" / "regression.csv",
                categorical_features=[],
                table_name=None,
                group_name="test_group",
                filename="regression"
            )
            split = data_split.get_split(0)
            X_train, X_test, y_train, y_test = split.get_train_test()

            # Create workflow
            output_dir = tmp_path / "results"
            algorithm_names = ["linear"]
            feature_names = ["x", "y"]
            workflow_attributes = workflow_attributes or {
                "model": algorithm_config["linear"].instantiate(),
            }

            return workflow_class(
                evaluation_manager=evaluation_manager, 
                X_train=X_train, 
                X_test=X_test, 
                y_train=y_train, 
                y_test=y_test, 
                output_dir=output_dir, 
                algorithm_names=algorithm_names, 
                feature_names=feature_names, 
                workflow_attributes=workflow_attributes
            )
    return _create_workflow


class TestWorkflow:
    """Test class for the Workflow base class."""
    def test_init(self, workflow_factory, tmp_path):
        workflow = workflow_factory(WorkflowSubclass)
        assert isinstance(workflow, Workflow)
        assert isinstance(workflow.evaluation_manager, EvaluationManager)
        assert isinstance(workflow.X_train, pd.DataFrame)
        assert workflow.X_train.attrs["is_test"] == False
        assert isinstance(workflow.X_test, pd.DataFrame)
        assert workflow.X_test.attrs["is_test"] == True
        assert isinstance(workflow.y_train, pd.Series)
        assert workflow.y_train.attrs["is_test"] == False
        assert isinstance(workflow.y_test, pd.Series)
        assert workflow.y_test.attrs["is_test"] == True
        assert workflow.output_dir == tmp_path / "results"
        assert workflow.algorithm_names == ["linear"]
        assert workflow.feature_names == ["x", "y"]
        assert isinstance(workflow.model, LinearRegression)

    def test_init_invalid_workflow(self, workflow_factory):
        with pytest.raises(
            TypeError, 
            match="Can't instantiate abstract class InvalidWorkflowSubclass"
        ):
            workflow_factory(InvalidWorkflowSubclass)

    def test_unpack_attributes_model_kwargs(self, workflow_factory):
        """Test that model_kwargs are unpacked as attributes correctly."""
        workflow = workflow_factory(WorkflowSubclass)
        assert hasattr(workflow, "model")
        assert isinstance(workflow.model, LinearRegression)

        workflow2 = workflow_factory(WorkflowSubclass, workflow_attributes={
            "metrics": ["MAE", "CCC"],
            "kf": 3,
            "string_attribute": "test"
            }
        )
        assert workflow2.metrics == ["MAE", "CCC"]
        assert workflow2.kf == 3
        assert workflow2.string_attribute == "test"

    def test_abstract_method(self):
        """Test that Workflow raises TypeError if instantiated without implementing 'workflow'."""
        evaluation_manager = MagicMock()
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_train = pd.Series()
        y_test = pd.Series()
        output_dir = "output"
        algorithm_names = []
        feature_names = []
        workflow_attributes = {}

        with pytest.raises(TypeError):
            Workflow(
                evaluation_manager=evaluation_manager, 
                X_train=X_train, 
                X_test=X_test, 
                y_train=y_train, 
                y_test=y_test, 
                output_dir=output_dir, 
                algorithm_names=algorithm_names, 
                feature_names=feature_names, 
                workflow_attributes=workflow_attributes
            )

    def test_not_implemented_error_direct(self):
        """Test that directly calling the base class 'workflow' method raises NotImplementedError."""
        
        class TempWorkflow(Workflow):
            def workflow(self, X_train, X_test, y_train, y_test, output_dir, feature_names):
                super().workflow(X_train, X_test, y_train, y_test, output_dir, feature_names)

        evaluation_manager = MagicMock()
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_train = pd.Series()
        y_test = pd.Series()
        output_dir = "output"
        algorithm_names = []
        feature_names = []
        workflow_attributes = {}

        temp_workflow = TempWorkflow(
            evaluation_manager, X_train, X_test, y_train, y_test, output_dir, 
            algorithm_names, feature_names, workflow_attributes
        )
        
        with pytest.raises(
            NotImplementedError, 
            match="Subclass must implement the workflow method."
            ):
            temp_workflow.workflow(X_train, X_test, y_train, y_test, output_dir, feature_names)
