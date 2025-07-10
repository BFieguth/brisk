from unittest.mock import MagicMock
import importlib
import inspect
        
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from brisk.evaluation.evaluation_manager import EvaluationManager
from brisk.training.workflow import Workflow

class WorkflowSubclass(Workflow): # pragma: no cover
    def workflow(self): 
        """Mock workflow implementation for testing."""
        self.evaluate_model(
            self.model, self.X_train, self.y_train, "MAE", "evaluate_model"
        )


class InvalidWorkflowSubclass(Workflow): # pragma: no cover
    def not_a_workflow(self):
        pass


@pytest.fixture
def workflow_factory(mock_brisk_project, tmp_path):
    """Fixture to initialize a Workflow instance with mock data."""
    def _create_workflow(workflow_class=WorkflowSubclass, workflow_attributes=None):
        # Load algorithm config    
        algorithm_file = tmp_path / "algorithms.py"
        algorithm_spec = importlib.util.spec_from_file_location(
            "algorithms", str(algorithm_file)
        )
        algorithm_module = importlib.util.module_from_spec(algorithm_spec)
        algorithm_spec.loader.exec_module(algorithm_module)
        algorithm_config = algorithm_module.ALGORITHM_CONFIG

        # Load metric config    
        metric_file = tmp_path / "metric.py"
        metric_spec = importlib.util.spec_from_file_location(
            "metric", str(metric_file)
        )
        metric_module = importlib.util.module_from_spec(metric_spec)
        metric_spec.loader.exec_module(metric_module)
        metric_config = metric_module.METRIC_CONFIG

        # Setup EvaluationManager
        evaluator = EvaluationManager(
            algorithm_config,
            metric_config,
            tmp_path / "results",
            {
                "num_features": 2,
                "num_samples": 4
            },
            logger=None
        )

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
            group_name=None,
            filename=None
        )
        X_train, X_test, y_train, y_test = data_split.get_train_test()

        # Create workflow
        output_dir = tmp_path / "results"
        algorithm_names = ["linear"]
        feature_names = ["x", "y"]
        workflow_attributes = workflow_attributes or {
            "model": algorithm_config["linear"].instantiate(),
        }

        return workflow_class(
            evaluator=evaluator, 
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
        assert isinstance(workflow.evaluator, EvaluationManager)
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

    def test_getattr_error_message(self, workflow_factory):
        """Test that the AttributeError message lists available attributes."""
        workflow = workflow_factory(WorkflowSubclass)
        with pytest.raises(AttributeError) as exc:
            _ = workflow.not_found
        assert "Available attributes are:" in str(exc.value)

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
        evaluator = MagicMock()
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
                evaluator=evaluator, 
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
        workflow_attributes = {}

        temp_workflow = TempWorkflow(
            evaluator, X_train, X_test, y_train, y_test, output_dir, 
            algorithm_names, feature_names, workflow_attributes
        )
        
        with pytest.raises(
            NotImplementedError, 
            match="Subclass must implement the workflow method."
            ):
            temp_workflow.workflow()

    def test_workflow_delegates_all_evaluation_manager_methods(self):
        """Test that Workflow has delegating methods for all public EvaluationManager methods."""
        # Define utility methods to exclude from delegation requirements
        utility_methods = {'save_model', 'load_model'}
        
        # Get all public methods from EvaluationManager
        evaluation_methods = []
        for name, _ in inspect.getmembers(
            EvaluationManager, predicate=inspect.isfunction
        ):
            if not name.startswith('_') and name not in utility_methods:
                evaluation_methods.append(name)
        
        # Check each evaluation method exists in Workflow
        missing_methods = []
        for method_name in evaluation_methods:
            if not hasattr(Workflow, method_name):
                missing_methods.append(method_name)

        assert not missing_methods, \
            "Workflow is missing delegating methods for EvaluationManager " \
            f"methods: {missing_methods}"

    def test_workflow_method_signatures_match_evaluation_manager(self):
        """Test that Workflow delegating methods have matching signatures with EvaluationManager methods."""
        # Define utility methods to exclude
        utility_methods = {'save_model', 'load_model'}
        
        # Get evaluation methods
        evaluation_methods = []
        for name, _ in inspect.getmembers(EvaluationManager, predicate=inspect.isfunction):
            if not name.startswith('_') and name not in utility_methods:
                evaluation_methods.append(name)
        
        # Check signatures match for each method
        signature_mismatches = []
        for method_name in evaluation_methods:
            if not hasattr(Workflow, method_name):
                continue  # Skip if method doesn't exist (covered by other test)
                
            eval_method = getattr(EvaluationManager, method_name)
            workflow_method = getattr(Workflow, method_name)
            
            eval_sig = inspect.signature(eval_method)
            workflow_sig = inspect.signature(workflow_method)
            
            # Compare parameters (excluding 'self')
            eval_params = list(eval_sig.parameters.values())[1:]
            workflow_params = list(workflow_sig.parameters.values())[1:]
            
            # Check parameter count
            if len(eval_params) != len(workflow_params):
                signature_mismatches.append(
                    f"{method_name}: parameter count mismatch - "
                    f"EvaluationManager has {len(eval_params)}, Workflow has {len(workflow_params)}"
                )
                continue
            
            # Check each parameter matches exactly
            for eval_param, workflow_param in zip(eval_params, workflow_params):
                if eval_param.name != workflow_param.name:
                    signature_mismatches.append(
                        f"{method_name}: parameter name mismatch - "
                        f"'{eval_param.name}' vs '{workflow_param.name}'"
                    )
                if eval_param.annotation != workflow_param.annotation:
                    signature_mismatches.append(
                        f"{method_name}.{eval_param.name}: type annotation mismatch - "
                        f"{eval_param.annotation} vs {workflow_param.annotation}"
                    )
                if eval_param.default != workflow_param.default:
                    signature_mismatches.append(
                        f"{method_name}.{eval_param.name}: default value mismatch - "
                        f"{eval_param.default} vs {workflow_param.default}"
                    )
                if eval_param.kind != workflow_param.kind:
                    signature_mismatches.append(
                        f"{method_name}.{eval_param.name}: parameter kind mismatch - "
                        f"{eval_param.kind} vs {workflow_param.kind}"
                    )
        
        assert not signature_mismatches, \
            f"Method signature mismatches found:\n" + "\n".join(signature_mismatches)
