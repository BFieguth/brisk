"""The Workflow base class for defining training and evaluation steps.

Specific workflows (e.g., regression, classification) should 
inherit from this class and implement the abstract `workflow` method to define 
the steps for model training, evaluation, and reporting.

Exports:
    - Workflow: A base class for building machine learning workflows.
"""

import abc
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
import sklearn.base as base

from brisk.evaluation.evaluation_manager import EvaluationManager

class Workflow(abc.ABC):
    """Base class for machine learning workflows.

    Attributes:
        evaluator (Evaluator): An object for evaluating models.
        X_train (pd.DataFrame): The training feature data.
        X_test (pd.DataFrame): The test feature data.
        y_train (pd.Series): The training target data.
        y_test (pd.Series): The test target data.
        output_dir (str): The directory where results are saved.
        method_names (List[str]): Names of the methods/models used.
        model1, model2, ...: The models passed to the workflow.
    """
    def __init__(
        self,
        evaluator: EvaluationManager, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series, 
        output_dir: str, 
        method_names: List[str], 
        feature_names: List[str],
        model_kwargs: Dict[str, Any],
        workflow_config = None
    ):
        self.evaluator = evaluator
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.output_dir = output_dir
        self.method_names = method_names
        self.feature_names = feature_names
        self._unpack_attributes(model_kwargs)
        if workflow_config:
            self._unpack_attributes(workflow_config)

    def __getattr__(self, name: str) -> None:
        if hasattr(self.evaluator, name):
            return getattr(self.evaluator, name)

        available_attrs = ", ".join(self.__dict__.keys())
        raise AttributeError(
            f"'{name}' not found. Available attributes are: {available_attrs}"
            )

    def _unpack_attributes(self, config: Dict[str, Any]) -> None:
        """
        Unpacks the key-value pairs from the config dictionary and sets them as 
        attributes of the instance.

        Args:
            config (Dict[str, Any]): The configuration dictionary to unpack.
        """
        for key, model in config.items():
            setattr(self, key, model)
 
    @abc.abstractmethod
    def workflow(self) -> None:
        raise NotImplementedError("Subclass must implement the workflow method.")
    
    # Delegate EvalutationManager
    def evaluate_model( # pragma: no cover
        self, 
        model: base.BaseEstimator, 
        X: pd.DataFrame, 
        y: pd.Series, 
        metrics: List[str], 
        filename: str
    ) -> None:
        """Evaluate the given model on the provided metrics and save the results.

        Args:
            model (BaseEstimator): The trained machine learning model to evaluate.
            X (pd.DataFrame): The feature data to use for evaluation.
            y (pd.Series): The target data to use for evaluation.
            metrics (List[str]): A list of metric names to calculate.
            filename (str): The name of the output file without extension.

        Returns:
            None
        """
        return self.evaluator.evaluate_model(model, X, y, metrics, filename)

    def evaluate_model_cv( # pragma: no cover
        self, 
        model: base.BaseEstimator, 
        X: pd.DataFrame, 
        y: pd.Series, 
        metrics: List[str], 
        filename: str, 
        cv: int = 5
    ) -> None:
        """Evaluate the model using cross-validation and save the scores.

        Args:
            model (BaseEstimator): The machine learning model to evaluate.
            X (pd.DataFrame): The feature data to use for evaluation.
            y (pd.Series): The target data to use for evaluation.
            metrics (List[str]): A list of metric names to calculate.
            filename (str): The name of the output file without extension.
            cv (int): The number of cross-validation folds. Defaults to 5.

        Returns:
            None
        """
        return self.evaluator.evaluate_model_cv(
            model, X, y, metrics, filename, cv=cv
            )

    def compare_models( # pragma: no cover
        self, 
        *models: base.BaseEstimator,
        X: pd.DataFrame, 
        y: pd.Series, 
        metrics: List[str], 
        filename: str, 
        calculate_diff: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models based on the provided metrics.

        Args:
            models: A variable number of model instances to evaluate.
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target data.
            metrics (List[str]): A list of metric names to calculate.
            filename (str): The name of the output file without extension.
            calculate_diff (bool): Whether to compute the difference between 
                models for each metric. Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary containing the metric results 
                for each model.
        """
        return self.evaluator.compare_models(*models, X=X, y=y, metrics=metrics, filename=filename, calculate_diff=calculate_diff)

    def plot_pred_vs_obs( # pragma: no cover
        self, 
        model: base.BaseEstimator, 
        X: pd.DataFrame, 
        y_true: pd.Series, 
        filename: str
    ) -> None:
        """Plot predicted vs. observed values and save the plot.

        Args:
            model (BaseEstimator): The trained machine learning model.
            X (pd.DataFrame): The feature data.
            y_true (pd.Series): The true target values.
            filename (str): The name of the output PNG file (without extension).

        Returns:
            None
        """
        return self.evaluator.plot_pred_vs_obs(model, X, y_true, filename)

    def plot_learning_curve( # pragma: no cover
        self, 
        model: base.BaseEstimator, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        cv: int = 5, 
        num_repeats: int = 1, 
        n_jobs: int = -1,
        metric: str = 'neg_mean_absolute_error', 
        filename: str = 'learning_curve'
    ) -> None:
        """
        Plot a learning curve for the given model and save the plot.

        Args:
            model (BaseEstimator): The machine learning model to evaluate.
            X_train (pd.DataFrame): The input features of the training set.
            y_train (pd.Series): The target values of the training set.
            cv (int): Number of cross-validation folds. Defaults to 5.
            num_repeats (int): Number of times to repeat the cross-validation. Defaults to 1.
            metric (str): The scoring metric to use. Defaults to 'neg_mean_absolute_error'.
            filename (str): The name of the output PNG file (without extension).

        Returns:
            None
        """
        return self.evaluator.plot_learning_curve(
            model, X_train, y_train, cv=cv, num_repeats=num_repeats, 
            n_jobs=n_jobs, metric=metric, filename=filename
            )

    def plot_feature_importance( # pragma: no cover
        self, 
        model: base.BaseEstimator, 
        X: pd.DataFrame, 
        y: pd.Series, 
        filter: Union[int, float], 
        feature_names: List[str], 
        filename: str, 
        metric: str, 
        num_rep: int
    ) -> None:
        """Plot the feature importance for the model and save the plot.

        Args:
            model (BaseEstimator): The machine learning model to evaluate.
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target data.
            filter (Union[int, float]): The number of features or the threshold 
                to filter features by importance.
            feature_names (List[str]): A list of feature names corresponding to 
                the columns in X.
            filename (str): The name of the output PNG file (without extension).
            metric (str): The metric to use for evaluation.
            num_rep (int): The number of repetitions for calculating importance.

        Returns:
            None
        """
        return self.evaluator.plot_feature_importance(
            model, X, y, filter, feature_names, filename, metric, num_rep
            )

    def plot_residuals( # pragma: no cover
        self, 
        model: base.BaseEstimator, 
        X: pd.DataFrame, 
        y: pd.Series, 
        filename: str
    ) -> None:
        """Plot the residuals of the model and save the plot.

        Args:
            model (BaseEstimator): The trained machine learning model.
            X (pd.DataFrame): The feature data.
            y (pd.Series): The true target values.
            filename (str): The name of the output PNG file (without extension).

        Returns:
            None
        """
        return self.evaluator.plot_residuals(model, X, y, filename)

    def plot_model_comparison( # pragma: no cover
        self, 
        *models: base.BaseEstimator, 
        X: pd.DataFrame, 
        y: pd.Series, 
        metric: str, 
        filename: str
    ) -> None:
        """Plot a comparison of multiple models based on the specified metric.

        Args:
            models: A variable number of model instances to evaluate.
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target data.
            metric (str): The metric to evaluate and plot.
            filename (str): The name of the output PNG file (without extension).

        Returns:
            None
        """
        return self.evaluator.plot_model_comparison(*models, X=X, y=y, metric=metric, filename=filename)

    def hyperparameter_tuning( # pragma: no cover
        self, 
        model: base.BaseEstimator, 
        method: str, 
        method_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        scorer: str, 
        kf: int, 
        num_rep: int, 
        n_jobs: int, 
        plot_results: bool = False
    ) -> base.BaseEstimator:
        """Perform hyperparameter tuning using grid or random search.

        Args:
            model (BaseEstimator): The model to be tuned.
            method (str): The search method to use ('grid' or 'random').
            method_name (str): The name of the method for which the hyperparameter 
                grid is being used.
            X_train (pd.DataFrame): The training data.
            y_train (pd.Series): The target values for training.
            scorer (str): The scoring metric to use.
            kf (int): Number of splits for cross-validation.
            num_rep (int): Number of repetitions for cross-validation.
            n_jobs (int): Number of parallel jobs to run.
            plot_results (bool): Whether to plot the performance of 
                hyperparameters. Defaults to False.

        Returns:
            BaseEstimator: The tuned model.
        """
        return self.evaluator.hyperparameter_tuning(
            model, method, method_name, X_train, y_train, scorer, 
            kf, num_rep, n_jobs, plot_results=plot_results
            )

    def confusion_matrix( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        filename: str
    ) -> None:
        """
        Generate a confusion matrix for a given model and dataset, 
        and save the results to a JSON file.

        Args:
            model (Any): Trained classification model with a `predict` method.
            X (np.ndarray): Input feature.
            y (np.ndarray): Target feature.
            filename (str): Path to save the confusion matrix as a JSON file.

        Returns:
            None
        """
        return self.evaluator.confusion_matrix(
            model, X, y, filename
        )
    
    def plot_confusion_heatmap( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        filename: str
    ) -> None:
        """
        Generate and save a heatmap of the confusion matrix for a given model and dataset.

        Args:
            model (Any): Trained classification model with a `predict` method.
            X (np.ndarray): Input features.
            y (np.ndarray): Target labels.
            filename (str): Path to save the confusion matrix heatmap image.
            labels (Optional[list]): List of class labels for display on the heatmap axes.

        Returns:
            None
        """
        return self.evaluator.plot_confusion_heatmap(
            model, X, y, filename
        )

    def plot_roc_curve( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        filename: str
    ) -> None:
        """
        Generate and save a ROC curve with AUC for a given binary classification model and dataset.

        Args:
            model (Any): Trained binary classification model with a `predict_proba` method.
            X (np.ndarray): Input features.
            y (np.ndarray): True binary labels.
            filename (str): Path to save the ROC curve image.

        Returns:
            None
        """
        return self.evaluator.plot_roc_curve(
            model, X, y, filename
        )

    def plot_precision_recall_curve( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        filename: str
    ) -> None:
        """
        Generate and save a Precision-Recall (PR) curve with Average Precision (AP) for a 
        given binary classification model and dataset.

        Args:
            model (Any): Trained binary classification model.
            X (np.ndarray): Input features.
            y (np.ndarray): True binary labels.
            filename (str): Path to save the PR curve image.

        Returns:
            None
        """
        return self.evaluator.plot_precision_recall_curve(
            model, X, y, filename
        )
