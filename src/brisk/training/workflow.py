"""The Workflow base class for defining training and evaluation steps.

This module provides the base Workflow class that defines the interface for
machine learning workflows. Specific workflows should inherit from this class
and implement the abstract `workflow` method. This class delegates the
EvaluationManager for model evaluation and visualization.
"""

import abc
from typing import List, Dict, Any, Union, Optional

import numpy as np
import pandas as pd
from sklearn import base

from brisk.evaluation import evaluation_manager as eval_manager

class Workflow(abc.ABC):
    """Base class for machine learning workflows. Delegates EvaluationManager.

    Parameters
    ----------
    evaluator : EvaluationManager
        Manager for model evaluation and visualization
    X_train : DataFrame
        Training feature data
    X_test : DataFrame
        Test feature data
    y_train : Series
        Training target data
    y_test : Series
        Test target data
    output_dir : str
        Directory where results will be saved
    algorithm_names : list of str
        Names of the algorithms used
    feature_names : list of str
        Names of the features
    workflow_attributes : dict
        Additional attributes to be unpacked into the workflow

    Attributes
    ----------
    evaluator : EvaluationManager
        Manager for model evaluation
    X_train : DataFrame
        Training feature data
    X_test : DataFrame
        Test feature data
    y_train : Series
        Training target data
    y_test : Series
        Test target data
    output_dir : str
        Output directory path
    algorithm_names : list of str
        Algorithm names
    feature_names : list of str
        Feature names
    model1, model2, ... : BaseEstimator
        Models unpacked from workflow_attributes
    """
    def __init__(
        self,
        evaluation_manager: eval_manager.EvaluationManager,
        X_train: pd.DataFrame, # pylint: disable=C0103
        X_test: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str,
        algorithm_names: List[str],
        feature_names: List[str],
        workflow_attributes: Dict[str, Any]
    ):
        self.evaluation_manager = evaluation_manager
        self.X_train = X_train # pylint: disable=C0103
        self.X_train.attrs["is_test"] = False
        self.X_test = X_test # pylint: disable=C0103
        self.X_test.attrs["is_test"] = True
        self.y_train = y_train
        self.y_train.attrs["is_test"] = False
        self.y_test = y_test
        self.y_test.attrs["is_test"] = True
        self.output_dir = output_dir
        self.algorithm_names = algorithm_names
        self.feature_names = feature_names
        self._unpack_attributes(workflow_attributes)

    def _unpack_attributes(self, config: Dict[str, Any]) -> None:
        """Unpack configuration dictionary into instance attributes.

        Parameters
        ----------
        config : dict
            Configuration dictionary to unpack
        """
        for key, model in config.items():
            setattr(self, key, model)

    @abc.abstractmethod
    def workflow(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str,
        feature_names: List[str],
    ) -> None:
        """Define the workflow steps for training and evaluation.
        
        This method should be implemented by subclasses to define the specific
        steps for training models, evaluation, and any other workflow logic.
        
        Parameters
        ----------
        X_train : DataFrame
            Training feature data with is_test=False attribute
        X_test : DataFrame
            Test feature data with is_test=True attribute
        y_train : Series
            Training target data with is_test=False attribute
        y_test : Series
            Test target data with is_test=True attribute
        output_dir : str
            Directory where results should be saved
        feature_names : list of str
            Names of the features in the datasets
            
        Notes
        -----
        Models and additional variables from workflow_attributes are available
        as instance attributes (e.g., self.model, self.user_variable, etc.)
        """
        raise NotImplementedError(
            "Subclass must implement the workflow method."
        )

    def run(self) -> None:
        """Call the workflow method with the processed parameters.
        """
        self.workflow(
            self.X_train, self.X_test, self.y_train, self.y_test,
            self.output_dir, self.feature_names,
        )

    # Interface to call Evaluators registered to EvaluationManager
    def evaluate_model( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str
    ) -> None:
        """Evaluate model on specified metrics and save results.

        Parameters
        ----------
        model : BaseEstimator
            Trained model to evaluate
        X : DataFrame
            Feature data
        y : Series
            Target data
        metrics : list of str
            Names of metrics to calculate
        filename : strm
            Output filename (without extension)
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_evaluate_model"
        )
        return evaluator.evaluate(model, X, y, metrics, filename)

    def evaluate_model_cv( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str,
        cv: int = 5
    ) -> None:
        """Evaluate model using cross-validation.

        Parameters
        ----------
        model : BaseEstimator
            Model to evaluate
        X : DataFrame
            Feature data
        y : Series
            Target data
        metrics : list of str
            Names of metrics to calculate
        filename : str
            Output filename (without extension)
        cv : int, optional
            Number of cross-validation folds, by default 5
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_evaluate_model_cv"
        )
        return evaluator.evaluate(model, X, y, metrics, filename, cv)

    def compare_models( # pragma: no cover
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str,
        calculate_diff: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models using specified metrics.

        Parameters
        ----------
        *models : BaseEstimator
            Models to compare
        X : DataFrame
            Feature data
        y : Series
            Target data
        metrics : list of str
            Names of metrics to calculate
        filename : str
            Output filename (without extension)
        calculate_diff : bool, optional
            Whether to compute differences between models, by default False

        Returns
        -------
        dict
            Nested dictionary containing metric results for each model
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_compare_models"
        )
        return evaluator.evaluate(
            *models, X=X, y=y, metrics=metrics, filename=filename,
            calculate_diff=calculate_diff
        )

    def plot_pred_vs_obs( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y_true: pd.Series,
        filename: str
    ) -> None:
        """Plot predicted vs. observed values and save the plot.

        Parameters
        ----------
        model (BaseEstimator):
            The trained model.
        X (pd.DataFrame):
            The input features.
        y_true (pd.Series):
            The true target values.
        filename (str):
            The name of the output file (without extension).
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_pred_vs_obs"
        )
        return evaluator.plot(model, X, y_true, filename)

    def plot_learning_curve( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X_train: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        filename: str = "learning_curve",
        cv: int = 5,
        num_repeats: int = 1,
        n_jobs: int = -1,
        metric: str = "neg_mean_absolute_error"
    ) -> None:
        """Plot learning curves showing model performance vs training size.

        Parameters
        ----------
        model : BaseEstimator
            Model to evaluate
        X_train : DataFrame
            Training features
        y_train : Series
            Training target values
        filename : str, optional
            Name for output file, by default "learning_curve"
        cv : int, optional
            Number of cross-validation folds, by default 5
        num_repeats : int, optional
            Number of times to repeat CV, by default 1
        n_jobs : int, optional
            Number of parallel jobs, by default -1
        metric : str, optional
            Scoring metric to use, by default "neg_mean_absolute_error"
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_learning_curve"
        )
        return evaluator.plot(
            model, X_train, y_train, filename=filename, cv=cv,
            num_repeats=num_repeats, n_jobs=n_jobs, metric=metric
        )

    def plot_feature_importance( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        threshold: Union[int, float],
        feature_names: List[str],
        filename: str,
        metric: str,
        num_rep: int
    ) -> None:
        """Plot the feature importance for the model and save the plot.

        Parameters
        ----------
        model (BaseEstimator):
            The model to evaluate.

        X (pd.DataFrame):
            The input features.

        y (pd.Series):
            The target data.

        threshold (Union[int, float]):
            The number of features or the threshold to filter features by
            importance.

        feature_names (List[str]):
            A list of feature names corresponding to the columns in X.

        filename (str):
            The name of the output file (without extension).

        metric (str):
            The metric to use for evaluation.

        num_rep (int):
            The number of repetitions for calculating importance.
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_feature_importance"
        )
        return evaluator.plot(
            model, X, y, threshold, feature_names, filename, metric, num_rep
        )

    def plot_residuals( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        filename: str,
        add_fit_line: bool = False
    ) -> None:
        """Plot the residuals of the model and save the plot.

        Parameters
        ----------
        model (BaseEstimator):
            The trained model.

        X (pd.DataFrame):
            The input features.

        y (pd.Series):
            The true target values.

        filename (str):
            The name of the output file (without extension).

        add_fit_line (bool):
            Whether to add a line of best fit to the plot.
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_residuals"
        )
        return evaluator.plot(model, X, y, filename, add_fit_line=add_fit_line)

    def plot_model_comparison( # pragma: no cover
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metric: str,
        filename: str
    ) -> None:
        """Plot a comparison of multiple models based on the specified metric.

        Parameters
        ----------
        models:
            A variable number of model instances to evaluate.

        X (pd.DataFrame):
            The input features.

        y (pd.Series):
            The target data.

        metric (str):
            The metric to evaluate and plot.

        filename (str):
            The name of the output file (without extension).
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_model_comparison"
        )
        return evaluator.plot(
            *models, X=X, y=y, metric=metric, filename=filename
        )

    def hyperparameter_tuning( # pragma: no cover
        self,
        model: base.BaseEstimator,
        method: str,
        X_train: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        scorer: str,
        kf: int,
        num_rep: int,
        n_jobs: int,
        plot_results: bool = False
    ) -> base.BaseEstimator:
        """Perform hyperparameter tuning using grid or random search.

        Parameters
        ----------
        model : BaseEstimator
            Model to tune
        method : {'grid', 'random'}
            Search method to use
        X_train : DataFrame
            Training data
        y_train : Series
            Training targets
        scorer : str
            Scoring metric
        kf : int
            Number of cross-validation splits
        num_rep : int
            Number of CV repetitions
        n_jobs : int
            Number of parallel jobs
        plot_results : bool, optional
            Whether to plot hyperparameter performance, by default False

        Returns
        -------
        BaseEstimator
            Tuned model
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_hyperparameter_tuning"
        )
        return evaluator.evaluate(
            model, method, X_train, y_train, scorer,
            kf, num_rep, n_jobs, plot_results=plot_results
        )

    def confusion_matrix( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str
    ) -> None:
        """Generate and save a confusion matrix.

        Parameters
        ----------
        model : Any
            Trained classification model with predict method

        X : ndarray
            The input features.

        y : ndarray
            The true target values.

        filename : str
            The name of the output file (without extension).
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_confusion_matrix"
        )
        return evaluator.evaluate(model, X, y, filename)

    def plot_confusion_heatmap( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str
    ) -> None:
        """Plot a heatmap of the confusion matrix for a model.

        Parameters
        ----------
        model (Any):
            The trained classification model with a `predict` method.

        X (np.ndarray):
            The input features.

        y (np.ndarray):
            The target labels.

        filename (str):
            The path to save the confusion matrix heatmap image.
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_confusion_heatmap"
        )
        return evaluator.plot(model, X, y, filename)

    def plot_roc_curve( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str,
        pos_label: Optional[int] = 1
    ) -> None:
        """Plot a reciever operator curve with area under the curve.

        Parameters
        ----------
        model (Any):
            The trained binary classification model.

        X (np.ndarray):
            The input features.

        y (np.ndarray):
            The true binary labels.

        filename (str):
            The path to save the ROC curve image.

        pos_label (Optional[int]):
            The label of the positive class.
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_roc_curve"
        )
        return evaluator.plot(model, X, y, filename, pos_label)

    def plot_precision_recall_curve( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str,
        pos_label: Optional[int] = 1
    ) -> None:
        """Plot a precision-recall curve with average precision.

        Parameters
        ----------
        model (Any):
            The trained binary classification model.

        X (np.ndarray):
            The input features.

        y (np.ndarray):
            The true binary labels.

        filename (str):
            The path to save the plot.

        pos_label (int):
            The label of the positive class.
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_precision_recall_curve"
        )
        return evaluator.plot(
            model, X, y, filename, pos_label
        )

    def save_model(self, model: base.BaseEstimator, filename: str) -> None: #pragma: no cover
        """Save model to pickle file.

        Parameters
        ----------
        model (BaseEstimator): 
            The model to save.

        filename (str): 
            The name for the output file (without extension).
        """
        self.evaluation_manager.save_model(model, filename)

    def load_model(self, filepath: str) -> base.BaseEstimator: #pragma: no cover
        """Load model from pickle file.

        Parameters
        ----------
        filepath : str
            Path to saved model file

        Returns
        -------
        BaseEstimator
            Loaded model

        Raises
        ------
        FileNotFoundError
            If model file does not exist
        """
        self.evaluation_manager.load_model(filepath)
