import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from brisk.training.TrainingManager import TrainingManager
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper
from brisk.data.DataManager import DataManager

@pytest.fixture
def setup_training_manager():
    """
    Setup for the TrainingManager instance.
    Mocks required objects for unit testing.
    """
    model_mock = mock.MagicMock()
    wrapper_mock = mock.MagicMock(spec=AlgorithmWrapper)
    wrapper_mock.instantiate.return_value = model_mock 

    method_config = {
        'linear': wrapper_mock 
    }
    
    metric_config = mock.MagicMock()
    data_manager = mock.MagicMock(spec=DataManager)
    methods = ['linear']
    data_paths = [('dataset.csv', None)]

    X_train = pd.DataFrame(
        np.random.rand(10, 3), columns=['feature1', 'feature2', 'feature3']
        )
    X_test = pd.DataFrame(
        np.random.rand(5, 3), columns=['feature1', 'feature2', 'feature3']
        )
    y_train = pd.Series(np.random.rand(10))
    y_test = pd.Series(np.random.rand(5))
    scaler_mock = mock.MagicMock()
    feature_names = ['feature1', 'feature2', 'feature3']

    data_manager.split.return_value = (
        X_train, X_test, y_train, y_test, scaler_mock, feature_names
        )
    data_manager.categorical_features = []

    return TrainingManager(
        method_config, metric_config, data_manager, methods, data_paths
    )


class TestTrainingManager:
    @mock.patch("brisk.Workflow.workflow")
    def test_consume_experiments(self, mock_workflow_method, setup_training_manager):
        """
        Test that experiments are consumed as they are processed.
        """
        mock_workflow_method.__name__ = "mock_workflow"
        setup_training_manager.data_splits["dataset.csv"].scaler = StandardScaler()

        tm = setup_training_manager
        assert len(tm.experiments) == 1

        with mock.patch("os.makedirs", wraps=os.makedirs):
            tm.run_experiments(mock_workflow_method, create_report=False)
        
        assert len(tm.experiments) == 0

    @mock.patch("brisk.Workflow.workflow")
    def test_run_experiments_calls_workflow_function(self, mock_workflow_method, setup_training_manager):
        """
        Test that the user-defined workflow function is called correctly for each experiment.
        """
        mock_workflow_method.__name__ = "mock_workflow"
        setup_training_manager.data_splits["dataset.csv"].scaler = StandardScaler()
        
        tm = setup_training_manager

        with mock.patch("os.makedirs", wraps=os.makedirs):
            tm.run_experiments(mock_workflow_method, create_report=False)

        assert mock_workflow_method.call_count == 1

    def test_invalid_method_in_experiment(self, setup_training_manager):
        """
        Test that ValueError is raised when a method not in the method_config is used.
        """
        invalid_methods = ['not_a_model']
        
        with pytest.raises(ValueError, match="The following methods are not included in the configuration: \['not_a_model'\]"):
            tm = TrainingManager(
                method_config=setup_training_manager.method_config,
                metric_config=setup_training_manager.metric_config,
                data_manager=setup_training_manager.DataManager,
                methods=invalid_methods,
                data_paths=setup_training_manager.data_paths,
            )
            