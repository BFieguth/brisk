"""Integration tests for multiple workflow functionality.

This module contains integration tests that verify the multiple workflow
feature works correctly across different scenarios.
"""

import pytest
import tempfile
import shutil
import importlib.util
import sys
from pathlib import Path
from unittest import mock

from brisk.training.training_manager import TrainingManager
from brisk.configuration.configuration_manager import ConfigurationManager
from brisk.evaluation.metric_manager import MetricManager
from brisk.configuration.experiment_group import ExperimentGroup
from brisk.configuration.algorithm_wrapper import AlgorithmWrapper
from brisk.training.workflow import Workflow


@pytest.fixture
def configuration(mock_brisk_project):
    """
    Create a configuration instance using the create_configuration function.
    """
    settings_file = mock_brisk_project / "settings.py"
    spec = importlib.util.spec_from_file_location(
        "settings", str(settings_file)
    )
    settings_module = importlib.util.module_from_spec(spec)
    sys.modules["settings"] = settings_module
    spec.loader.exec_module(settings_module)

    return settings_module.create_configuration()


@pytest.fixture  
def metric_config(mock_brisk_project):
    metrics_file = mock_brisk_project / "metrics.py"
    spec = importlib.util.spec_from_file_location("metrics", str(metrics_file))
    metrics_module = importlib.util.module_from_spec(spec)
    sys.modules["metrics"] = metrics_module
    spec.loader.exec_module(metrics_module)
    return metrics_module.METRIC_CONFIG


@pytest.fixture
def training_manager(configuration, metric_config):
    return TrainingManager(
        metric_config=metric_config,
        config_manager=configuration
    )


class TestMultipleWorkflows:
    """Test multiple workflow functionality across different scenarios."""

    def test_no_default_workflow(self, mock_brisk_project):

        group1 = ExperimentGroup(
            name="group1",
            description="Group without workflow",
            workflow=None,  
            datasets=["regression.csv"],
            workflow_args={},
            data_config={}
        )

        # This should raise an error since no workflow is specified
        with pytest.raises((AttributeError, ValueError, TypeError)):
            ConfigurationManager(
                experiment_groups=[group1],
                categorical_features={}
            )

    def test_only_default_workflow(self, mock_brisk_project, metric_config):
        group1 = ExperimentGroup(
            name="group1",
            description="Group with only default workflow",
            datasets=["regression.csv"],
            algorithms=["linear"],
            workflow="regression_workflow"
        )

        from brisk.configuration.configuration import Configuration
        config = Configuration(
            default_workflow="regression_workflow",
            default_algorithms=["linear"]
        )
        config.add_experiment_group(**group1.__dict__)
        
        config_manager = config.build()
        training_manager = TrainingManager(
            metric_config=metric_config,
            config_manager=config_manager
        )

        assert "regression_workflow" in training_manager.workflow_mapping
   

    def test_group_override_default(self, mock_brisk_project, mock_categorical_workflow, metric_config):
        from brisk.configuration.configuration import Configuration
        
        config = Configuration(
            default_workflow="regression_workflow",
            default_algorithms=["linear"]
        )
        

        config.add_experiment_group(
            name="special_group",
            description="Group that overrides default workflow",
            datasets=["regression.csv"],
            algorithms=["linear"],
            workflow="categorical_workflow"
        )
        
        config_manager = config.build()
        training_manager = TrainingManager(
            metric_config=metric_config,
            config_manager=config_manager
        )
        
        assert "categorical_workflow" in training_manager.workflow_mapping

        
        experiments = list(config_manager.experiment_queue)
        assert len(experiments) == 2 
        assert all(exp.workflow == "categorical_workflow" for exp in experiments)

    def test_group_with_default(self, training_manager, configuration):
        assert "regression_workflow" in training_manager.workflow_mapping

        experiments = list(configuration.experiment_queue)
        assert len(experiments) > 0
        assert all(e.workflow == "regression_workflow" for e in experiments)

    def test_multiple_groups_each_own_workflow(self, mock_brisk_project, mock_categorical_workflow, metric_config):
        from brisk.configuration.configuration import Configuration
        
        config = Configuration(
            default_workflow="regression_workflow",
            default_algorithms=["linear"]
        )
        

        config.add_experiment_group(
            name="regression_group",
            description="Group using regression workflow",
            datasets=["regression.csv"],
            algorithms=["linear"],
            workflow="regression_workflow"
        )
        
        config.add_experiment_group(
            name="categorical_group", 
            description="Group using categorical workflow",
            datasets=["regression.csv"],
            algorithms=["ridge"],
            workflow="categorical_workflow"
        )
        
        config_manager = config.build()
        training_manager = TrainingManager(
            metric_config=metric_config,
            config_manager=config_manager
        )
        

        assert "regression_workflow" in training_manager.workflow_mapping
        assert "categorical_workflow" in training_manager.workflow_mapping
        
        experiments = list(config_manager.experiment_queue)
        assert len(experiments) == 4  
        
        workflow_assignments = {exp.group_name: exp.workflow for exp in experiments}
        assert workflow_assignments["regression_group"] == "regression_workflow"
        assert workflow_assignments["categorical_group"] == "categorical_workflow"

    def test_workflow_missing_mapping(self, mock_brisk_project, capfd):

        metrics_file = mock_brisk_project / "metrics.py"
        spec = importlib.util.spec_from_file_location("metrics", str(metrics_file))
        metrics_module = importlib.util.module_from_spec(spec)
        sys.modules["metrics"] = metrics_module
        spec.loader.exec_module(metrics_module)
        metric_config = metrics_module.METRIC_CONFIG

        group = ExperimentGroup(
            name="test_group",
            description="Test group",
            datasets=["regression.csv"],
            algorithms=["linear"],
            workflow="nonexistent_workflow"
        )

        from brisk.configuration.configuration import Configuration
        config = Configuration(
            default_workflow="regression_workflow",
            default_algorithms=["linear"]
        )
        config.add_experiment_group(**group.__dict__)
        
        config_manager = config.build()
        

        captured = capfd.readouterr()
        assert ("Error validating workflow" in captured.out or 
                "No module named 'workflows'" in captured.out)
        

        assert "nonexistent_workflow" in config_manager.workflow_map
        assert config_manager.workflow_map["nonexistent_workflow"] is None

     