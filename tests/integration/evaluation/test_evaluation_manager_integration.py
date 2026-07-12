"""Integration tests for EvaluationManager."""

import pytest
from unittest import mock
import pathlib
import shutil

from sklearn import linear_model
import numpy as np
import joblib

from brisk.evaluation import evaluation_manager
from brisk.adapters.sklearn import metric_adapter as metric_manager
from brisk.services import bundle, io, rerun
from brisk.configuration import project
from brisk.theme import plot_settings

# pylint: disable=W0621, C0103

@pytest.fixture()
def evaluator_templates():
    fixtures_dir = pathlib.Path(__file__).parent.parent.parent / "fixtures"
    return fixtures_dir / "evaluators"


@pytest.fixture()
def copy_evaluator_file(tmp_path, evaluator_templates):
    """Factory fixture to copy evaluator files to tmp_path."""
    def _copy(template_name: str) -> pathlib.Path:
        source = evaluator_templates / f"{template_name}.py"
        dest = tmp_path / "evaluators.py"
        shutil.copy2(source, dest)
        return dest

    return _copy


@pytest.fixture()
def mock_services(tmp_path):
    mock_logger = mock.Mock()
    mock_logger.logger = mock.Mock()
    mock_metadata = mock.Mock()
    mock_utility = mock.Mock()
    mock_utility.get_plot_settings.return_value = plot_settings.PlotSettings()
    mock_reporting = mock.Mock()
    with project.ProjectRootContext(tmp_path):
        mock_rerun = rerun.RerunService("rerun")
    mock_io = io.IOService("io", tmp_path, tmp_path)
    mock_io.register_services({
        "rerun": mock_rerun,
        "logging": mock_logger
    })

    return bundle.ServiceBundle(
        io=mock_io,
        logger=mock_logger,
        metadata=mock_metadata,
        utility=mock_utility,
        reporting=mock_reporting,
        rerun=mock_rerun
    )


@pytest.fixture()
def sample_metric_manager():
    return metric_manager.SklearnMetricManager()


@pytest.fixture()
def evaluation_mgr(sample_metric_manager):
    return evaluation_manager.EvaluationManager(sample_metric_manager)


@pytest.fixture()
def sample_model():
    model = linear_model.LogisticRegression(random_state=42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    return model


@pytest.mark.integration
class TestEvaluationManagerIntegration:
    """Integration tests for the EvaluationManager class."""

    def test_register_custom_evaluators(
        self, tmp_path, evaluation_mgr, mock_services, copy_evaluator_file
    ):
        """Test successful registration of custom evaluators."""
        with project.ProjectRootContext(tmp_path):
            copy_evaluator_file("custom_single")

            evaluation_mgr.set_services(mock_services)
            evaluation_mgr.initialize_evaluators()

        assert "custom_evaluator" in evaluation_mgr.registry.evaluators

    def test_register_custom_evaluators_no_file(
        self, tmp_path, evaluation_mgr, mock_services
    ):
        """
        Test that initialize_evaluators works without evaluators.py file.
        Custom evaluators are optional - should not raise an error.
        """
        with project.ProjectRootContext(tmp_path):
            evaluation_mgr.set_services(mock_services)
            # Should not raise - custom evaluators are optional
            evaluation_mgr.initialize_evaluators()
            # Built-in evaluators should still be registered
            assert "brisk_evaluate_model" in evaluation_mgr.registry.evaluators

    def test_check_unregistered_evaluators_all_registered(
        self, tmp_path, evaluation_mgr, mock_services, copy_evaluator_file
    ):
        """Test no warnings when all evaluators are registered."""
        with project.ProjectRootContext(tmp_path):
            copy_evaluator_file("all_registered")

            evaluation_mgr.set_services(mock_services)
            evaluation_mgr.initialize_evaluators()

        mock_services.logger.logger.warning.assert_not_called()

    def test_check_unregistered_evaluators_one_unregistered(
        self, tmp_path, evaluation_mgr, mock_services, copy_evaluator_file
    ):
        """Test warning is logged for one unregistered evaluator."""
        with project.ProjectRootContext(tmp_path):
            copy_evaluator_file("unregistered_one")

            evaluation_mgr.set_services(mock_services)
            evaluation_mgr.initialize_evaluators()

        mock_services.logger.logger.warning.assert_called_once()

        warning_message = mock_services.logger.logger.warning.call_args[0][0]
        assert "UnregisteredEvaluator" in warning_message
        assert "unregistered" in warning_message.lower()

    def test_check_unregistered_evaluators_two_unregistered(
        self, tmp_path, evaluation_mgr, mock_services, copy_evaluator_file
    ):
        """Test warnings are logged for multiple unregistered evaluators."""
        with project.ProjectRootContext(tmp_path):
            copy_evaluator_file("unregistered_two")

            evaluation_mgr.set_services(mock_services)
            evaluation_mgr.initialize_evaluators()

        assert mock_services.logger.logger.warning.call_count == 2

        warning_messages = [
            call[0][0]
            for call in mock_services.logger.logger.warning.call_args_list
        ]
        assert any("UnregisteredOne" in msg for msg in warning_messages)
        assert any("UnregisteredTwo" in msg for msg in warning_messages)

    def test_initialize_evaluators(
        self, tmp_path, evaluation_mgr, mock_services, copy_evaluator_file
    ):
        """Test initialization of built-in and custom evaluators."""
        with project.ProjectRootContext(tmp_path):
            copy_evaluator_file("unregistered_two")

            evaluation_mgr.set_services(mock_services)
            evaluation_mgr.initialize_evaluators()

        assert len(evaluation_mgr.registry.evaluators) > 0

    def test_set_evaluator_services(
        self, tmp_path, evaluation_mgr, mock_services, copy_evaluator_file
    ):
        """Test that services are set for all evaluators."""
        with project.ProjectRootContext(tmp_path):
            copy_evaluator_file("custom_single")

            evaluation_mgr.set_services(mock_services)
            evaluation_mgr.initialize_evaluators()
            evaluation_mgr.set_evaluator_services()

        for evaluator in evaluation_mgr.registry.evaluators.values():
            assert evaluator.services is not None

    def test_save_model(
        self,
        tmp_path,
        evaluation_mgr,
        mock_services,
        sample_model
    ):
        """Test model saving functionality."""
        evaluation_mgr.set_services(mock_services)
        evaluation_mgr.set_output_dir(str(tmp_path))

        mock_metadata_dict = {"model_type": "LogisticRegression", "params": {}}
        mock_services.metadata.get_model.return_value = mock_metadata_dict

        evaluation_mgr.save_model(sample_model, "test_model")

        expected_path = tmp_path / "test_model.pkl"
        assert expected_path.exists()

    def test_save_model_metadata(
        self, tmp_path, evaluation_mgr, mock_services, sample_model
    ):
        """Test that model metadata is properly saved with the model."""
        evaluation_mgr.set_services(mock_services)
        evaluation_mgr.set_output_dir(str(tmp_path))

        expected_metadata = {
            "model_type": "LogisticRegression",
            "params": {"C": 1.0, "random_state": 42},
            "timestamp": "2024-01-01T12:00:00"
        }
        mock_services.metadata.get_model.return_value = expected_metadata

        evaluation_mgr.save_model(sample_model, "test_model")

        # Load and verify metadata was saved
        model_package = joblib.load(tmp_path / "test_model.pkl")

        assert "model" in model_package
        assert "metadata" in model_package
        assert model_package["metadata"] == expected_metadata
        model = model_package["model"]
        assert isinstance(model, linear_model.LogisticRegression)

    def test_load_model(
        self, tmp_path, evaluation_mgr, mock_services, sample_model
    ):
        """Test model loading functionality."""
        evaluation_mgr.set_services(mock_services)
        evaluation_mgr.set_output_dir(str(tmp_path))

        mock_services.metadata.get_model.return_value = {
            "model_type": "LogisticRegression"
        }
        model_path = tmp_path / "test_model.pkl"
        evaluation_mgr.save_model(sample_model, "test_model")

        loaded_package = evaluation_mgr.load_model(str(model_path))

        assert isinstance(
            loaded_package["model"], linear_model.LogisticRegression
        )

    def test_load_model_missing_file(
        self, tmp_path, evaluation_mgr, mock_services
    ):
        """Test that FileNotFoundError is raised for missing model file."""
        evaluation_mgr.set_services(mock_services)

        non_existent_path = tmp_path / "nonexistent_model.pkl"

        with pytest.raises(FileNotFoundError, match="No model found at"):
            evaluation_mgr.load_model(str(non_existent_path))
