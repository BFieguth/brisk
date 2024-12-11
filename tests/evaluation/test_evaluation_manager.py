import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge 
from sklearn.metrics import mean_absolute_error, make_scorer
from pathlib import Path
from brisk.evaluation.evaluation_manager import EvaluationManager
from brisk.utility.algorithm_wrapper import AlgorithmWrapper


@pytest.fixture
def sample_data():
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature{i}' for i in range(5)])
    y = pd.Series(np.random.rand(100))
    return X, y


@pytest.fixture
def sample_data_cat():
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature{i}' for i in range(5)])
    y = pd.Series(np.random.choice([0, 1], size=100))
    return X, y


@pytest.fixture
def eval_manager(tmpdir):
    algorithm_config = [
        AlgorithmWrapper(
            name="random_forest",
            display_name="Random Forest",
            algorithm_class=RandomForestRegressor,
            default_params={'n_jobs': 1},
            hyperparam_grid={
                'n_estimators': list(range(20, 160, 20))
            }
        ),
    ]
    metric_config = MagicMock()
    metric_config.get_name.return_value = "Mean Absolute Error"
    metric_config.get_metric.return_value = mean_absolute_error
    metric_config.get_scorer.return_value = make_scorer(mean_absolute_error)
    return EvaluationManager(
        algorithm_config, metric_config, output_dir=str(tmpdir), 
        split_metadata={}, logger=MagicMock()
    )


@pytest.fixture
def model(sample_data):
    X, y = sample_data
    model = RandomForestRegressor(n_estimators=10, n_jobs=1)
    model.fit(X, y) 
    return model


@pytest.fixture
def model_classifier(sample_data_cat):
    X, y = sample_data_cat
    model = RandomForestClassifier(n_estimators=10, n_jobs=1)
    model.fit(X, y) 
    return model


@pytest.fixture()
def model2(sample_data):
    X, y = sample_data
    model2 = Ridge(alpha=0.1)
    model2.fit(X, y)
    return model2


@pytest.fixture
def model_with_proba():
    """Mock binary classification model with predict_proba."""
    class MockModel:
        def predict_proba(self, X):
            return np.random.rand(X.shape[0], 2)
    return MockModel()


class TestEvaluationManager:
    def test_evaluate_model(self, eval_manager, model, sample_data, tmpdir):
        X, y = sample_data
        filename = tmpdir.join("evaluation_result")
        metrics = ["mean_absolute_error"]

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_to_json") as mock_save_json:

            eval_manager.evaluate_model(model, X, y, metrics, filename)

            mock_save_json.assert_called_once()
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            eval_manager.logger.info.assert_called()

    def test_evaluate_model_cv(self, eval_manager, model, sample_data, tmpdir):
        X, y = sample_data
        filename = tmpdir.join("cv_evaluation_result")
        metrics = ["mean_absolute_error"]

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_to_json") as mock_save_json:
            
            eval_manager.evaluate_model_cv(model, X, y, metrics, filename)
           
            mock_save_json.assert_called_once()
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            eval_manager.logger.info.assert_called()

    def test_compare_models(self, eval_manager, model, model2, sample_data, tmpdir):
        X, y = sample_data
        filename = tmpdir.join("comparison_result")
        metrics = ["mean_absolute_error"]

        with patch("os.makedirs") as mock_makedirs, \
            patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_to_json") as mock_save_json:
        
            result = eval_manager.compare_models(model, model2, X=X, y=y, metrics=metrics, filename=filename)
           
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_json.assert_called_once()
            eval_manager.logger.info.assert_called()
            assert "RandomForestRegressor" in result

    def test_plot_pred_vs_obs(self, eval_manager, model, sample_data, tmpdir):
        X, y = sample_data
        filename = tmpdir.join("pred_vs_obs_plot")

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:
           
            eval_manager.plot_pred_vs_obs(model, X, y, filename)
           
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_plot_learning_curve(self, eval_manager, model, sample_data, tmpdir):
        X, y = sample_data
        filename = tmpdir.join("learning_curve")

        with patch("os.makedirs") as mock_makedirs, \
            patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:
            
            eval_manager.plot_learning_curve(
                model, X, y, filename=filename, n_jobs=1
                )
            
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_plot_feature_importance(self, eval_manager, model, sample_data, tmpdir):
        X, y = sample_data
        feature_names = [f'feature{i}' for i in range(X.shape[1])]
        filename = tmpdir.join("feature_importance")

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:
            
            eval_manager.plot_feature_importance(
                model, X, y, threshold=3, feature_names=feature_names,
                filename=filename, metric="mean_absolute_error", num_rep=1
            )
            
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_plot_residuals(self, eval_manager, model, sample_data, tmpdir):
        X, y = sample_data
        filename = tmpdir.join("residuals_plot")

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:
            
            eval_manager.plot_residuals(model, X, y, filename)
            
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_hyperparameter_tuning(self, eval_manager, model, sample_data):
        X, y = sample_data

        with patch("brisk.evaluation.evaluation_manager.EvaluationManager._plot_hyperparameter_performance") as mock_plot:

            tuned_model = eval_manager.hyperparameter_tuning(
                model=model, method="grid", algorithm_name="random_forest", X_train=X, y_train=y,
                scorer="neg_mean_squared_error", kf=3, num_rep=1, n_jobs=1, plot_results=True
            )
           
            mock_plot.assert_called_once()
            assert tuned_model is not None
            eval_manager.logger.info.assert_called()

    def test_save_and_load_model(self, eval_manager, model, tmpdir):
        filename = tmpdir.join("saved_model")
        eval_manager.output_dir = tmpdir

        with patch("os.makedirs") as mock_makedirs:
            eval_manager.save_model(model, filename)
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            
        assert Path(f"{filename}.pkl").exists()

        loaded_model = eval_manager.load_model(f"{filename}.pkl")
        assert isinstance(loaded_model, RandomForestRegressor)

    def test_plot_model_comparison(self, eval_manager, model, model2,sample_data):
        X, y = sample_data
        models = [model, model2]
        metric = "mean_absolute_error"
        filename = "model_comparison"

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager.plot_model_comparison(*models, X=X, y=y, metric=metric, filename=filename)
            
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_plot_hyperparameter_performance(self, eval_manager):
        param_grid = {"n_estimators": [10, 50, 100]}
        search_result = MagicMock()
        search_result.cv_results_ = {'mean_test_score': [0.8, 0.85, 0.87]}
        algorithm_name = "random_forest"
        metadata = {"test": "data"}

        with patch("brisk.evaluation.evaluation_manager.EvaluationManager._plot_1d_performance") as mock_plot_1d, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._plot_3d_surface") as mock_plot_3d:

            eval_manager._plot_hyperparameter_performance(param_grid, search_result, algorithm_name, metadata)
            
            mock_plot_1d.assert_called_once_with(
                param_values=param_grid["n_estimators"],
                mean_test_score=search_result.cv_results_['mean_test_score'],
                param_name="n_estimators",
                algorithm_name=algorithm_name,
                metadata=metadata
            )
            mock_plot_3d.assert_not_called()

    def test_plot_1d_performance(self, eval_manager):
        param_values = [10, 50, 100]
        mean_test_score = [0.8, 0.85, 0.87]
        param_name = "n_estimators"
        algorithm_name = "random_forest"
        metadata = {"test": "data"}

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager._plot_1d_performance(param_values, mean_test_score, param_name, algorithm_name, metadata)
            
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_plot_3d_surface(self, eval_manager):
        param_grid = {"max_depth": [5, 10, 15], "min_samples_split": [2, 4]}
        search_result = MagicMock()
        search_result.cv_results_ = {'mean_test_score': np.array([0.8, 0.82, 0.85, 0.83, 0.86, 0.87]).reshape(3, 2)}
        param_names = ["max_depth", "min_samples_split"]
        algorithm_name = "random_forest"
        metadata = {"test": "data"}

        with patch("os.makedirs") as mock_makedirs, \
             patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager._plot_3d_surface(param_grid, search_result, param_names, algorithm_name, metadata)
            
            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_confusion_matrix(self, eval_manager, model_classifier, sample_data_cat):
        """Test the confusion_matrix method of EvaluationManager."""
        X, y = sample_data_cat
        filename = "confusion_matrix"

        with patch("os.makedirs") as mock_makedirs, \
            patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_to_json") as mock_save_json:

            eval_manager.confusion_matrix(model_classifier, X, y, filename)

            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_json.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_plot_confusion_heatmap(self, eval_manager, model_classifier, sample_data_cat):
        """Test the plot_confusion_heatmap method of EvaluationManager."""
        X, y = sample_data_cat
        filename = "confusion_heatmap"

        with patch("os.makedirs") as mock_makedirs, \
            patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager.plot_confusion_heatmap(model_classifier, X, y, filename)

            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_plot_roc_curve(self, eval_manager, model_with_proba, sample_data_cat):
        """Test the plot_roc_curve method of EvaluationManager."""
        X, y = sample_data_cat

        filename = "roc_curve"

        with patch("os.makedirs") as mock_makedirs, \
            patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager.plot_roc_curve(model_with_proba, X, y, filename)

            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_plot_precision_recall_curve(self, eval_manager, model_with_proba, sample_data_cat):
        """Test the plot_precision_recall_curve method of EvaluationManager."""
        X, y = sample_data_cat
        filename = "precision_recall_curve"

        with patch("os.makedirs") as mock_makedirs, \
            patch("brisk.evaluation.evaluation_manager.EvaluationManager._save_plot") as mock_save_plot:

            eval_manager.plot_precision_recall_curve(model_with_proba, X, y, filename)

            mock_makedirs.assert_called_once_with(eval_manager.output_dir, exist_ok=True)
            mock_save_plot.assert_called_once()
            eval_manager.logger.info.assert_called()

    def test_save_to_json(self, eval_manager, tmpdir):
        data = {"score": 0.95}
        metadata = {"experiment": "test"}
        output_path = tmpdir.join("test_result.json")

        with patch("builtins.open", mock_open()) as mock_file:
            eval_manager._save_to_json(data, str(output_path), metadata)

            # Verify that the file was opened and json was written
            mock_file.assert_called_once_with(
                str(output_path), "w", encoding="utf-8"
                )
            
            # Check that metadata was added to the data dictionary
            data_with_metadata = data.copy()
            data_with_metadata["_metadata"] = metadata

            # Retrieve the actual JSON written by combining all `write` calls
            written_data = "".join(
                call.args[0] for call in mock_file().write.call_args_list
                )

            # Convert JSON string back to a dictionary for comparison
            written_data_dict = json.loads(written_data)
            assert written_data_dict == data_with_metadata

    def test_save_to_json_io_error(self, eval_manager, tmpdir):
        data = {"score": 0.95}
        metadata = {"experiment": "test"}
        output_path = tmpdir.join("test_result.json")

        # Simulate IOError
        with patch("builtins.open", side_effect=IOError("File write error")) as mock_file:
            eval_manager._save_to_json(data, str(output_path), metadata)

            # Verify that logger.info was called due to IOError
            eval_manager.logger.info.assert_called_once_with(
                f"Failed to save JSON to {str(output_path)}: File write error"
            )

    def test_save_plot(self, eval_manager, tmpdir):
        output_path = tmpdir.join("test_plot.png")
        metadata = {"experiment": "test"}

        # Create a simple plot to test saving
        plt.plot([1, 2, 3], [4, 5, 6])

        with patch("matplotlib.pyplot.savefig") as mock_savefig, \
             patch("matplotlib.pyplot.close") as mock_close:
            eval_manager._save_plot(str(output_path), metadata)

            # Assert savefig was called with the correct parameters
            mock_savefig.assert_called_once_with(
                str(output_path), format="png", metadata=metadata
                )
            mock_close.assert_called_once()

    def test_save_plot_io_error(self, eval_manager, tmpdir):
        output_path = tmpdir.join("test_plot.png")
        metadata = {"experiment": "test"}

        plt.plot([1, 2, 3], [4, 5, 6])

        # Simulate IOError in plt.savefig
        with patch("matplotlib.pyplot.savefig", side_effect=IOError("File write error")):
            eval_manager._save_plot(str(output_path), metadata)

            # Verify that logger.info was called due to IOError
            eval_manager.logger.info.assert_called_once_with(
                f"Failed to save plot to {str(output_path)}: File write error"
            )
