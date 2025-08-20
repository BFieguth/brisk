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

from brisk.evaluation.evaluation_manager import EvaluationManager

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
        evaluator: EvaluationManager,
        X_train: pd.DataFrame, # pylint: disable=C0103
        X_test: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str,
        algorithm_names: List[str],
        feature_names: List[str],
        workflow_attributes: Dict[str, Any]
    ):
        self.evaluator = evaluator
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

    def __getattr__(self, name: str) -> None:
        if hasattr(self.evaluator, name):
            return getattr(self.evaluator, name)

        available_attrs = ", ".join(self.__dict__.keys())
        raise AttributeError(
            f"'{name}' not found. Available attributes are: {available_attrs}"
            )

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
    def workflow(self) -> None:
        raise NotImplementedError(
            "Subclass must implement the workflow method."
        )

    # Delegate EvalutationManager
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
        filename : str
            Output filename (without extension)
        """
        return self.evaluator.evaluate_model(model, X, y, metrics, filename)

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
        return self.evaluator.evaluate_model_cv(
            model, X, y, metrics, filename, cv=cv
            )

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
        return self.evaluator.compare_models(
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
        return self.evaluator.plot_pred_vs_obs(model, X, y_true, filename)

    def plot_learning_curve( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X_train: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        cv: int = 5,
        num_repeats: int = 1,
        n_jobs: int = -1,
        metric: str = "neg_mean_absolute_error",
        filename: str = "learning_curve"
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
        cv : int, optional
            Number of cross-validation folds, by default 5
        num_repeats : int, optional
            Number of times to repeat CV, by default 1
        n_jobs : int, optional
            Number of parallel jobs, by default -1
        metric : str, optional
            Scoring metric to use, by default "neg_mean_absolute_error"
        filename : str, optional
            Name for output file, by default "learning_curve"
        """
        return self.evaluator.plot_learning_curve(
            model, X_train, y_train, cv=cv, num_repeats=num_repeats,
            n_jobs=n_jobs, metric=metric, filename=filename
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
        return self.evaluator.plot_feature_importance(
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
        return self.evaluator.plot_residuals(
            model, X, y, filename, add_fit_line=add_fit_line
        )

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
        return self.evaluator.plot_model_comparison(
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
        return self.evaluator.hyperparameter_tuning(
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
        return self.evaluator.confusion_matrix(
            model, X, y, filename
        )

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
        return self.evaluator.plot_confusion_heatmap(
            model, X, y, filename
        )

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
        return self.evaluator.plot_roc_curve(
            model, X, y, filename, pos_label
        )

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
        return self.evaluator.plot_precision_recall_curve(
            model, X, y, filename, pos_label
        )
