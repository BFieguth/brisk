"""Provides methods for model evaluation and visualization.

This module defines the EvaluationManager class, which provides methods for
evaluating models, generating plots, and comparing models. These methods are
used when building a training workflow.
"""

import copy
import datetime
import inspect
import itertools
import json
import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn

from sklearn import base
from sklearn import ensemble
from sklearn import inspection
import sklearn.model_selection as model_select
import sklearn.metrics as sk_metrics
from sklearn import tree

from brisk.configuration import algorithm_wrapper
from brisk.theme import theme
from brisk.evaluation import metric_manager
matplotlib.use("Agg")


class EvaluationManager:
    """A class for evaluating machine learning models and plotting results.

    This class provides methods for model evaluation, including calculating
    metrics, generating plots, comparing models, and hyperparameter tuning. It
    is designed to be used within a Workflow instance.

    Parameters
    ----------
    algorithm_config : AlgorithmCollection
        Configuration for algorithms.
    metric_config : MetricManager
        Configuration for evaluation metrics.
    output_dir : str
        Directory to save results.
    split_metadata : Dict[str, Any]
        Metadata to include in metric calculations.
    logger : Optional[logging.Logger]
        Logger instance to use.

    Attributes
    ----------
    algorithm_config : AlgorithmCollection
        Configuration for algorithms.
    metric_config : Any
        Configuration for evaluation metrics.
    output_dir : str
        Directory to save results.
    split_metadata : Dict[str, Any]
        Metadata to include in metric calculations.
    logger : Optional[logging.Logger]
        Logger instance to use.
    primary_color : str
        Color for primary elements.
    secondary_color : str
        Color for secondary elements.
    background_color : str
        Color for background elements.
    accent_color : str
        Color for accent elements.
    important_color : str
        Color for important elements.
    """
    def __init__(
        self,
        algorithm_config: algorithm_wrapper.AlgorithmCollection,
        metric_config: metric_manager.MetricManager,
        output_dir: str,
        split_metadata: Dict[str, Any],
        group_index_train: Dict[str, np.array] | None,
        group_index_test: Dict[str, np.array] | None,
        logger: Optional[logging.Logger]=None
    ):
        self.algorithm_config = algorithm_config
        self.metric_config = copy.deepcopy(metric_config)
        self.metric_config.set_split_metadata(split_metadata)
        self.output_dir = output_dir
        self.group_index_train = group_index_train
        self.group_index_test = group_index_test
        if group_index_train is not None and group_index_test is not None:
            self.data_has_groups = True
        else:
            self.data_has_groups = False
        self.logger = logger

        self.primary_color = "#0074D9" # Celtic Blue
        self.secondary_color = "#07004D" # Federal Blue
        self.background_color = "#C4E0F9" # Columbia Blue
        self.accent_color = "#00A878" # Jade
        self.important_color = "#B95F89" # Mulberry

    # Evaluation Tools
    def evaluate_model(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str
    ) -> None:
        """Evaluate a model on the provided metrics and save the results.

        Parameters
        ----------
        model (BaseEstimator):
            The trained model to evaluate.
        X (pd.DataFrame):
            The input features.
        y (pd.Series):
            The target data.
        metrics (List[str]):
            A list of metrics to calculate.
        filename (str):
            The name of the output file without extension.
        """
        predictions = model.predict(X)
        results = self._calc_evaluate_model(predictions, y, metrics)
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        metadata = self._get_metadata(model, is_test=X.attrs["is_test"])
        self._save_to_json(results, output_path, metadata)

        scores_log = "\n".join([
            f"{metric}: {score:.4f}"
            if isinstance(score, (int, float))
            else f"{metric}: {score}"
            for metric, score in results.items()
            if metric != "_metadata"
            ]
        )
        self.logger.info(
            "Model evaluation results:\n%s\nSaved to '%s'.",
            scores_log, output_path
        )

    def _calc_evaluate_model(
        self,
        predictions: Dict[str, Any],
        y_true: pd.Series,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Calculate the evaluation results for a model.

        Parameters
        ----------
        predictions (Dict[str, Any]):
            The predictions of the model.
        y_true (pd.Series):
            The true target values.
        metrics (List[str]):
            A list of metrics to calculate.

        Returns:
        -------
        Dict[str, float]:
            A dictionary containing the evaluation results for each metric.
        """
        results = {}
        for metric_name in metrics:
            display_name = self.metric_config.get_name(metric_name)
            scorer = self.metric_config.get_metric(metric_name)
            if scorer is not None:
                score = scorer(y_true, predictions)
                results[display_name] = score
            else:
                self.logger.info(f"Scorer for {metric_name} not found.")
        return results

    def evaluate_model_cv(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str,
        cv: int = 5
    ) -> None:
        """Evaluate a model using cross-validation and save the scores.

        Parameters
        ----------
        model (BaseEstimator):
            The model to evaluate.
        X (pd.DataFrame):
            The input features.
        y (pd.Series):
            The target data.
        metrics (List[str]):
            A list of metrics to calculate.
        filename (str):
            The name of the output file without extension.
        cv (int):
            The number of cross-validation folds. Defaults to 5.
        """
        results = self._calc_evaluate_model_cv(model, X, y, metrics, cv)
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        metadata = self._get_metadata(model, is_test=X.attrs["is_test"])
        self._save_to_json(results, output_path, metadata)

        scores_log = "\n".join([
            f"{metric}: mean={res['mean_score']:.4f}, " # pylint: disable=W1405
            f"std_dev={res['std_dev']:.4f}" # pylint: disable=W1405
            for metric, res in results.items()
            if metric != "_metadata"
        ])
        self.logger.info(
            "Cross-validation results:\n%s\nSaved to '%s'.",
            scores_log, output_path
        )

    def _calc_evaluate_model_cv(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        cv: int = 5
    ) -> Dict[str, float]:
        """Calculate the cross-validation results for a model.

        Parameters
        ----------
        model (BaseEstimator):
            The model to evaluate.
        X (pd.DataFrame):
            The input features.
        y (pd.Series):
            The target data.
        metrics (List[str]):
            A list of metrics to calculate.
        cv (int):
            The number of cross-validation folds. Defaults to 5.
        """
        splitter, indices = self._get_cv_splitter(y, cv)
        results = {}
        for metric_name in metrics:
            display_name = self.metric_config.get_name(metric_name)
            scorer = self.metric_config.get_scorer(metric_name)
            if scorer is not None:
                scores = model_select.cross_val_score(
                    model, X, y, scoring=scorer, cv=splitter, groups=indices
                    )
                results[display_name] = {
                    "mean_score": scores.mean(),
                    "std_dev": scores.std(),
                    "all_scores": scores.tolist()
                }
            else:
                self.logger.info(f"Scorer for {metric_name} not found.")
        return results

    def compare_models(
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: List[str],
        filename: str,
        calculate_diff: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models using specified metrics.

        Parameters
        ----------
        *models : BaseEstimator
            Models to compare
        X : DataFrame
            Input features
        y : Series
            Target values
        metrics : list of str
            Names of metrics to calculate
        filename : str
            Name for output file (without extension)
        calculate_diff : bool, optional
            Whether to calculate differences between models, by default False
        """
        comparison_results = self._calc_compare_models(
            *models, X=X, y=y, metrics=metrics, calculate_diff=calculate_diff
        )
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        metadata = self._get_metadata(
            list(models), "compare_models", is_test=X.attrs["is_test"]
        )
        self._save_to_json(comparison_results, output_path, metadata)

        comparison_log = "\n".join([
            f"{model}: " +
            ", ".join(
                [f"{metric}: {score:.4f}"
                 if isinstance(score, (float, int, np.floating))
                 else f"{metric}: {score}" for metric, score in results.items()
                 if metric != "_metadata"]
                )
            for model, results in comparison_results.items()
            if model not in ["differences", "_metadata"]
        ])
        self.logger.info(
            "Model comparison results:\n%s\nSaved to '%s'.",
            comparison_log, output_path
        )

    def _calc_compare_models(
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: List[str],
        calculate_diff: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate the comparison results for multiple models.

        Parameters
        ----------
        *models : BaseEstimator
            Models to compare
        X : DataFrame
            Input features
        y : Series
            Target values
        metrics : list of str
            Names of metrics to calculate
        calculate_diff : bool, optional
            Whether to calculate differences between models, by default False

        Returns
        -------
        dict
            Nested dictionary containing metric scores for each model
        """
        comparison_results = {}

        if not models:
            raise ValueError("At least one model must be provided")

        model_names = []
        for model in models:
            wrapper = self._get_algo_wrapper(model.wrapper_name)
            model_names.append(wrapper.display_name)

        # Evaluate the model and collect results
        for model_name, model in zip(model_names, models):
            predictions = model.predict(X)
            results = {}

            for metric_name in metrics:
                scorer = self.metric_config.get_metric(metric_name)
                display_name = self.metric_config.get_name(metric_name)
                if scorer is not None:
                    score = scorer(y, predictions)
                    results[display_name] = score
                else:
                    self.logger.info(f"Scorer for {metric_name} not found.")

            comparison_results[model_name] = results

        # Calculate the difference between models for each metric
        if calculate_diff and len(models) > 1:
            comparison_results["differences"] = {}
            model_pairs = list(itertools.combinations(model_names, 2))

            for metric_name in metrics:
                display_name = self.metric_config.get_name(metric_name)
                comparison_results["differences"][display_name] = {}

                for model_a, model_b in model_pairs:
                    score_a = comparison_results[model_a][display_name]
                    score_b = comparison_results[model_b][display_name]
                    diff = score_b - score_a
                    comparison_results["differences"][display_name][
                        f"{model_b} - {model_a}"
                    ] = diff
        return comparison_results

    def plot_pred_vs_obs(
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
        prediction = model.predict(X)
        plot_data, max_range = self._calc_plot_pred_vs_obs(prediction, y_true)
        wrapper = self._get_algo_wrapper(model.wrapper_name)
        plot = (
            pn.ggplot(plot_data, pn.aes(x="Observed", y="Predicted")) +
            pn.geom_point(
                color="black", size=3, stroke=0.25, fill=self.primary_color
            ) +
            pn.geom_abline(
                slope=1, intercept=0, color=self.important_color,
                linetype="dashed"
            ) +
            pn.labs(
                x="Observed Values",
                y="Predicted Values",
                title=f"Predicted vs. Observed Values ({wrapper.display_name})"
            ) +
            pn.coord_fixed(
                xlim=[0, max_range],
                ylim=[0, max_range]
            ) +
            theme.brisk_theme()
        )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model, is_test=X.attrs["is_test"])
        self._save_plot(output_path, metadata, plot=plot)
        self.logger.info(
            "Predicted vs. Observed plot saved to '%s'.", output_path
        )

    def _calc_plot_pred_vs_obs(
        self,
        prediction: pd.Series,
        y_true: pd.Series,
    ) -> None:
        """Calculate the plot data for the predicted vs. observed values.

        Parameters
        ----------
        prediction (pd.Series):
            The predicted values.
        y_true (pd.Series):
            The true target values.

        Returns
        -------
        Tuple[pd.DataFrame, float]:
            A tuple containing the plot data and the maximum range of the plot.
        """
        plot_data = pd.DataFrame({
            "Observed": y_true,
            "Predicted": prediction
        })
        max_range = plot_data[["Observed", "Predicted"]].max().max()
        return plot_data, max_range

    def plot_learning_curve(
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
        results = self._calc_plot_learning_curve(
            model, X_train, y_train, cv, num_repeats, n_jobs, metric
        )
        # Create subplots
        _, axes = plt.subplots(1, 3, figsize=(16, 6))
        plt.rcParams.update({"font.size": 12})

        # Plot Learning Curve
        display_name = self.metric_config.get_name(metric)
        wrapper = self._get_algo_wrapper(model.wrapper_name)
        axes[0].set_title(
            f"Learning Curve ({wrapper.display_name})", fontsize=20
        )
        axes[0].set_xlabel("Training Examples", fontsize=12)
        axes[0].set_ylabel(display_name, fontsize=12)
        axes[0].grid()
        axes[0].fill_between(
            results["train_sizes"],
            results["train_scores_mean"] - results["train_scores_std"],
            results["train_scores_mean"] + results["train_scores_std"],
            alpha=0.1, color="r"
            )
        axes[0].fill_between(
            results["train_sizes"],
            results["test_scores_mean"] - results["test_scores_std"],
            results["test_scores_mean"] + results["test_scores_std"],
            alpha=0.1, color="g"
            )
        axes[0].plot(
            results["train_sizes"], results["train_scores_mean"], "o-",
            color="r", label="Training Score"
            )
        axes[0].plot(
            results["train_sizes"], results["test_scores_mean"], "o-",
            color="g", label="Cross-Validation Score"
            )
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(results["train_sizes"], results["fit_times_mean"], "o-")
        axes[1].fill_between(
            results["train_sizes"],
            results["fit_times_mean"] - results["fit_times_std"],
            results["fit_times_mean"] + results["fit_times_std"],
            alpha=0.1
            )
        axes[1].set_xlabel("Training Examples", fontsize=12)
        axes[1].set_ylabel("Fit Times", fontsize=12)
        axes[1].set_title("Scalability of the Model", fontsize=16)

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(
            results["fit_times_mean"], results["test_scores_mean"], "o-"
        )
        axes[2].fill_between(
            results["fit_times_mean"],
            results["test_scores_mean"] - results["test_scores_std"],
            results["test_scores_mean"] + results["test_scores_std"],
            alpha=0.1
            )
        axes[2].set_xlabel("Fit Times", fontsize=12)
        axes[2].set_ylabel(display_name, fontsize=12)
        axes[2].set_title("Performance of the Model", fontsize=16)

        plt.tight_layout()

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model, is_test=X_train.attrs["is_test"])
        self._save_plot(output_path, metadata)
        self.logger.info(f"Learning Curve plot saved to '{output_path}''.")

    def _calc_plot_learning_curve(
        self,
        model: base.BaseEstimator,
        X_train: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        cv: int = 5,
        num_repeats: Optional[int] = None,
        n_jobs: int = -1,
        metric: str = "neg_mean_absolute_error",
    ) -> Dict[str, float]:
        """Calculate the plot data for the learning curve.

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
            Number of times to repeat CV, by default None
        n_jobs : int, optional
            Number of parallel jobs, by default -1
        metric : str, optional
            Scoring metric to use, by default "neg_mean_absolute_error"

        Returns
        -------
        Dict[str, float]:
            A dictionary containing the learning curve data.
        """
        splitter, indices = self._get_cv_splitter(y_train, cv, num_repeats)
        results = {}
        scorer = self.metric_config.get_scorer(metric)

        # Generate learning curve data
        train_sizes, train_scores, test_scores, fit_times, _ = (
            model_select.learning_curve(
                model, X_train, y_train, cv=splitter, groups=indices,
                n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 5),
                return_times=True, scoring=scorer
            )
        )
        results["train_sizes"] = train_sizes
        # Calculate means and standard deviations
        results["train_scores_mean"] = np.mean(train_scores, axis=1)
        results["train_scores_std"] = np.std(train_scores, axis=1)
        results["test_scores_mean"] = np.mean(test_scores, axis=1)
        results["test_scores_std"] = np.std(test_scores, axis=1)
        results["fit_times_mean"] = np.mean(fit_times, axis=1)
        results["fit_times_std"] = np.std(fit_times, axis=1)
        return results

    def plot_feature_importance(
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
            The target data
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
        importance_data, plot_width, plot_height = (
            self._calc_plot_feature_importance(
                model, X, y, threshold, feature_names, metric, num_rep
            )
        )
        display_name = self.metric_config.get_name(metric)
        wrapper = self._get_algo_wrapper(model.wrapper_name)
        plot = (
            pn.ggplot(importance_data, pn.aes(x="Feature", y="Importance")) +
            pn.geom_bar(stat="identity", fill=self.primary_color) +
            pn.coord_flip() +
            pn.labs(
                x="Feature", y=f"Importance ({display_name})",
                title=f"Feature Importance ({wrapper.display_name})"
            ) +
            theme.brisk_theme()
        )
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model, is_test=X.attrs["is_test"])
        self._save_plot(output_path, metadata, plot, plot_height, plot_width)
        self.logger.info(
            "Feature Importance plot saved to '%s'.", output_path
        )

    def _calc_plot_feature_importance(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        threshold: Union[int, float],
        feature_names: List[str],
        metric: str,
        num_rep: int
    ) -> Tuple[pd.DataFrame, float, float]:
        """Calculate the plot data for the feature importance.

        Parameters
        ----------
        model (BaseEstimator):
            The model to evaluate.
        X (pd.DataFrame):
            The input features.
        y (pd.Series):
            The target data
        threshold (Union[int, float]):
            The number of features or the threshold to filter features by
            importance.
        feature_names (List[str]):
            A list of feature names corresponding to the columns in X.
        metric (str):
            The metric to use for evaluation.
        num_rep (int):
            The number of repetitions for calculating importance.

        Returns
        -------
        Tuple[pd.DataFrame, float, float]:
            A tuple containing the feature importance values, the width of the
            plot, and the height of the plot.
        """
        scorer = self.metric_config.get_scorer(metric)

        if isinstance(
            model, (
                tree.DecisionTreeRegressor, ensemble.RandomForestRegressor,
                ensemble.GradientBoostingRegressor)
            ):
            model.fit(X,y)
            importance = model.feature_importances_
        else:
            model.fit(X, y)
            results = inspection.permutation_importance(
                model, X=X, y=y, scoring=scorer, n_repeats=num_rep
                )
            importance = results.importances_mean

        if isinstance(threshold, int):
            sorted_indices = np.argsort(importance)[::-1]
            importance = importance[sorted_indices[:threshold]]
            feature_names = [
                feature_names[i] for i in sorted_indices[:threshold]
            ]
        elif isinstance(threshold, float):
            num_features = int(len(feature_names) * threshold)
            if num_features == 0:
                num_features = 1
            sorted_indices = np.argsort(importance)[::-1]
            importance = importance[sorted_indices[:num_features]]
            feature_names = [
                feature_names[i] for i in sorted_indices[:num_features]
            ]

        num_features = len(feature_names)
        size_per_feature = 0.1
        plot_width = max(
            8, size_per_feature * num_features
        )
        plot_height = max(
            6, size_per_feature * num_features * 0.75
        )
        importance_data = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })
        sorted_df = importance_data.sort_values("Importance")  # type: pd.DataFrame
        sorted_features = sorted_df["Feature"].tolist()  # pylint: disable=E1136
        importance_data["Feature"] = pd.Categorical(
            importance_data["Feature"],
            categories=sorted_features,
            ordered=True
        )
        return importance_data, plot_width, plot_height

    def plot_residuals(
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
        predictions = model.predict(X)
        plot_data = self._calc_plot_residuals(predictions, y)
        wrapper = self._get_algo_wrapper(model.wrapper_name)
        plot = (
            pn.ggplot(plot_data, pn.aes(
                x="Observed",
                y="Residual (Observed - Predicted)"
            )) +
            pn.geom_point(
                color="black", size=3, stroke=0.25, fill=self.primary_color
            ) +
            pn.geom_abline(
                slope=0, intercept=0, color=self.important_color,
                linetype="dashed", size=1.5
            ) +
            pn.ggtitle(f"Residuals ({wrapper.display_name})") +
            theme.brisk_theme()
        )

        if add_fit_line:
            fit = np.polyfit(
                plot_data["Observed"],
                plot_data["Residual (Observed - Predicted)"],
                1
            )
            fit_line = np.polyval(fit, plot_data["Observed"])
            plot += (
                pn.geom_line(
                    pn.aes(x="Observed", y=fit_line, group=1),
                    color=self.accent_color, size=1
                )
            )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model, is_test=X.attrs["is_test"])
        self._save_plot(output_path, metadata, plot)
        self.logger.info(
            "Residuals plot saved to '%s'.", output_path
        )

    def _calc_plot_residuals(
        self,
        predictions: pd.Series,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Calculate the residuals (observed - predicted).

        Parameters
        ----------
        predictions (pd.Series):
            The predicted values.
        y (pd.Series):
            The true target values.

        Returns
        -------
        pd.DataFrame:
            A dataframe containing the observed and residual values.
        """
        residuals = y - predictions
        plot_data = pd.DataFrame({
            "Observed": y,
            "Residual (Observed - Predicted)": residuals
        })
        return plot_data

    def plot_model_comparison(
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame,
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
        plot_data = self._calc_plot_model_comparison(
            *models, X=X, y=y, metric=metric
        )
        if plot_data is None:
            return

        display_name = self.metric_config.get_name(metric)
        plot = (
            pn.ggplot(plot_data, pn.aes(x="Model", y="Score")) +
            pn.geom_bar(stat="identity", fill=self.primary_color) +
            pn.geom_text(
                pn.aes(label="Score"), position=pn.position_stack(vjust=0.5),
                color="white", size=16
            ) +
            pn.ggtitle(f"Model Comparison on {display_name}") +
            pn.ylab(display_name) +
            theme.brisk_theme()
        )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(
            list(models), "plot_model_comparison", is_test=X.attrs["is_test"]
        )
        self._save_plot(output_path, metadata, plot)
        self.logger.info(
            "Model Comparison plot saved to '%s'.", output_path
        )
        plt.close()

    def _calc_plot_model_comparison(
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str
    ) -> pd.DataFrame:
        """Calculate the plot data for the model comparison.

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

        Returns
        -------
        pd.DataFrame:
            A dataframe containing the model names and their scores.
        """
        model_names = []
        for model in models:
            wrapper = self._get_algo_wrapper(model.wrapper_name)
            model_names.append(wrapper.display_name)

        metric_values = []

        scorer = self.metric_config.get_metric(metric)

        for model in models:
            predictions = model.predict(X)
            if scorer is not None:
                score = scorer(y, predictions)
                metric_values.append(round(score, 3))
            else:
                self.logger.info(f"Scorer for {metric} not found.")
                return None

        plot_data = pd.DataFrame({
            "Model": model_names,
            "Score": metric_values,
        })
        return plot_data

    def hyperparameter_tuning(
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
        model (BaseEstimator):
            The model to be tuned.

        method (str):
            The search method to use ("grid" or "random").

        X_train (pd.DataFrame):
            The training data.

        y_train (pd.Series):
            The target values for training.

        scorer (str):
            The scoring metric to use.

        kf (int):
            Number of splits for cross-validation.

        num_rep (int):
            Number of repetitions for cross-validation.

        n_jobs (int):
            Number of parallel jobs to run.

        plot_results (bool):
            Whether to plot the performance of hyperparameters. Defaults to
            False.

        Returns
        -------
        BaseEstimator:
            The tuned model.
        """
        algo_wrapper = self._get_algo_wrapper(model.wrapper_name)
        param_grid = algo_wrapper.get_hyperparam_grid()
        search_result = self._calc_hyperparameter_tuning(
            model, method, X_train, y_train, scorer, kf, num_rep, n_jobs,
            param_grid
        )
        tuned_model = algo_wrapper.instantiate_tuned(
            search_result.best_params_
        )
        tuned_model.fit(X_train, y_train)
        self.logger.info(
            "Hyperparameter optimization for %s complete.",
            model.__class__.__name__
            )

        if plot_results:
            metadata = self._get_metadata(
                model, is_test=X_train.attrs["is_test"]
            )
            self._plot_hyperparameter_performance(
                param_grid, search_result, algo_wrapper.name, metadata,
                algo_wrapper.display_name
            )
        return tuned_model

    def _calc_hyperparameter_tuning(
        self,
        model: base.BaseEstimator,
        method: str,
        X_train: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        scorer: str,
        kf: int,
        num_rep: int,
        n_jobs: int,
        param_grid: Dict[str, Any]
    ) -> base.BaseEstimator:
        """Perform hyperparameter tuning using grid or random search.

        Parameters
        ----------
        model (BaseEstimator):
            The model to be tuned.

        method (str):
            The search method to use ("grid" or "random").

        X_train (pd.DataFrame):
            The training data.

        y_train (pd.Series):
            The target values for training.

        scorer (str):
            The scoring metric to use.

        kf (int):
            Number of splits for cross-validation.

        num_rep (int):
            Number of repetitions for cross-validation.

        n_jobs (int):
            Number of parallel jobs to run.

        param_grid (Dict[str, Any]):
            The hyperparameter grid used for tuning.

        Returns
        -------
        BaseEstimator:
            The tuned model.
        """
        if method == "grid":
            searcher = model_select.GridSearchCV
        elif method == "random":
            searcher = model_select.RandomizedSearchCV
        else:
            raise ValueError(
                f"method must be one of (grid, random). {method} was entered."
                )

        self.logger.info(
            "Starting hyperparameter optimization for %s",
            model.__class__.__name__
            )
        score = self.metric_config.get_scorer(scorer)
        splitter, indices = self._get_cv_splitter(y_train, kf, num_rep)

        # The arguments for each sklearn searcher are different which is why the
        # first two arguments have no keywords. If adding another searcher make
        # sure the argument names do not conflict.
        search = searcher(
            model, param_grid, n_jobs=n_jobs, cv=splitter, scoring=score
        )
        search_result = search.fit(X_train, y_train, groups=indices)
        return search_result

    def _plot_hyperparameter_performance(
        self,
        param_grid: Dict[str, Any],
        search_result: Any,
        algorithm_name: str,
        metadata: Dict[str, Any],
        display_name: str
    ) -> None:
        """Plot the performance of hyperparameter tuning.

        Parameters
        ----------
        param_grid (Dict[str, Any]):
            The hyperparameter grid used for tuning.

        search_result (Any):
            The result from cross-validation during tuning.

        algorithm_name (str):
            The name of the algorithm.

        metadata (Dict[str, Any]):
            Metadata to be included with the plot.

        display_name (str):
            The name of the algorithm to use in the plot labels.
        """
        param_keys = list(param_grid.keys())

        if len(param_keys) == 0:
            return

        elif len(param_keys) == 1:
            self._plot_1d_performance(
                param_values=param_grid[param_keys[0]],
                mean_test_score=search_result.cv_results_["mean_test_score"],
                param_name=param_keys[0],
                algorithm_name=algorithm_name,
                metadata=metadata,
                display_name=display_name
            )
        elif len(param_keys) == 2:
            self._plot_3d_surface(
                param_grid=param_grid,
                search_result=search_result,
                param_names=param_keys,
                algorithm_name=algorithm_name,
                metadata=metadata,
                display_name=display_name
            )
        else:
            self.logger.info(
                "Higher dimensional visualization not implemented yet"
                )

    def _plot_1d_performance(
        self,
        param_values: List[Any],
        mean_test_score: List[float],
        param_name: str,
        algorithm_name: str,
        metadata: Dict[str, Any],
        display_name: str
    ) -> None:
        """Plot the performance of a single hyperparameter vs mean test score.

        Parameters
        ----------
        param_values (List[Any]):
            The values of the hyperparameter.

        mean_test_score (List[float]):
            The mean test scores for each hyperparameter value.

        param_name (str):
            The name of the hyperparameter.

        algorithm_name (str):
            The name of the algorithm.

        metadata (Dict[str, Any]):
            Metadata to be included with the plot.

        display_name (str):
            The name of the algorithm to use in the plot labels.
        """
        plot_data = pd.DataFrame({
            "Hyperparameter": param_values,
            "Mean Test Score": mean_test_score,
        })
        param_name = param_name.capitalize()
        title = f"Hyperparameter Performance: {display_name}"
        plot = (
            pn.ggplot(
                plot_data, pn.aes(x="Hyperparameter", y="Mean Test Score")
            ) +
            pn.geom_point(
                color="black", size=3, stroke=0.25, fill=self.primary_color
            ) +
            pn.geom_line(color=self.primary_color) +
            pn.ggtitle(title) +
            pn.xlab(param_name) +
            theme.brisk_theme()
        )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.output_dir, f"{algorithm_name}_hyperparam_{param_name}.png"
            )
        self._save_plot(output_path, metadata, plot)
        self.logger.info(
            "Hyperparameter performance plot saved to '%s'.", output_path
            )

    def _plot_3d_surface(
        self,
        param_grid: Dict[str, List[Any]],
        search_result: Any,
        param_names: List[str],
        algorithm_name: str,
        metadata: Dict[str, Any],
        display_name: str
    ) -> None:
        """Plot the performance of two hyperparameters vs mean test score.

        Parameters
        ----------
        param_grid (Dict[str, List[Any]]):
            The hyperparameter grid used for tuning.

        search_result (Any):
            The result from cross-validation during tuning.

        param_names (List[str]):
            The names of the two hyperparameters.

        algorithm_name (str):
            The name of the algorithm.

        metadata (Dict[str, Any]):
            Metadata to be included with the plot.

        display_name (str):
            The name of the algorithm to use in the plot labels.
        """
        X, Y, mean_test_score = self._calc_plot_3d_surface( # pylint: disable=C0103
            param_grid, search_result, param_names
        )
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, mean_test_score.T, cmap="viridis")
        ax.set_xlabel(param_names[0], fontsize=12)
        ax.set_ylabel(param_names[1], fontsize=12)
        ax.set_zlabel("Mean Test Score", fontsize=12)
        ax.set_title(
            f"Hyperparameter Performance: {display_name}", fontsize=16
        )
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.output_dir, f"{algorithm_name}_hyperparam_3Dplot.png"
            )
        self._save_plot(output_path, metadata)
        self.logger.info(
            "Hyperparameter performance plot saved to '%s'.", output_path
            )

    def _calc_plot_3d_surface(
        self,
        param_grid: Dict[str, List[Any]],
        search_result: Any,
        param_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the plot data for the 3D surface plot.

        Parameters
        ----------
        param_grid (Dict[str, List[Any]]):
            The hyperparameter grid used for tuning.

        search_result (Any):
            The result from cross-validation during tuning.

        param_names (List[str]):
            The names of the two hyperparameters.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing the X, Y ndarrays for the meshgrid, and mean test
            score values.
        """
        mean_test_score = search_result.cv_results_["mean_test_score"].reshape(
            len(param_grid[param_names[0]]),
            len(param_grid[param_names[1]])
        )
        X, Y = np.meshgrid( # pylint: disable=C0103
            param_grid[param_names[0]], param_grid[param_names[1]]
        )
        return X, Y, mean_test_score

    def confusion_matrix(
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
        prediction = model.predict(X)
        data = self._calc_confusion_matrix(prediction, y)
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        metadata = self._get_metadata(model, is_test=X.attrs["is_test"])
        self._save_to_json(data, output_path, metadata)

        header = " " * 10 + " ".join(
            f"{label:>10}" for label in data["labels"]
        ) + "\n"
        rows = [f"{label:>10} " + " ".join(f"{count:>10}" for count in row)
                for label, row in zip(data["labels"], data["confusion_matrix"])]
        table = header + "\n".join(rows)
        confusion_log = f"Confusion Matrix:\n{table}"
        self.logger.info(confusion_log)

    def _calc_confusion_matrix(
        self,
        prediction: pd.Series,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate a confusion matrix.

        Parameters
        ----------
        prediction (pd.Series):
            The predicted target values.
        y (np.ndarray):
            The true target values.

        Returns
        -------
        Dict[str, Any]:
            A dictionary containing the confusion matrix and labels.
        """
        labels = np.unique(y).tolist()
        cm = sk_metrics.confusion_matrix(y, prediction, labels=labels).tolist()
        data = {
            "confusion_matrix": cm,
            "labels": labels
            }
        return data

    def plot_confusion_heatmap(
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
        prediction = model.predict(X)
        plot_data = self._calc_plot_confusion_heatmap(prediction, y)
        wrapper = self._get_algo_wrapper(model.wrapper_name)
        plot = (
            pn.ggplot(plot_data, pn.aes(
                x="Predicted Label",
                y="True Label",
                fill="Percentage"
            )) +
            pn.geom_tile() +
            pn.geom_text(pn.aes(label="Label"), color="black") +
            pn.scale_fill_gradient( # pylint: disable=E1123
                low="white",
                high=self.primary_color,
                name="Percentage (%)",
                limits=(0, 100)
            ) +
            pn.ggtitle(f"Confusion Matrix Heatmap ({wrapper.display_name})") +
            theme.brisk_theme()
        )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model, is_test=X.attrs["is_test"])
        self._save_plot(output_path, metadata, plot)
        self.logger.info(f"Confusion matrix heatmap saved to {output_path}")

    def _calc_plot_confusion_heatmap(
        self,
        prediction: pd.Series,
        y: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate the plot data for the confusion matrix heatmap.

        Parameters
        ----------
        prediction (pd.Series):
            The predicted target values.
        y (np.ndarray):
            The true target values.

        Returns
        -------
        pd.DataFrame:
            A dataframe containing the confusion matrix heatmap data.
        """
        labels = np.unique(y).tolist()
        cm = sk_metrics.confusion_matrix(y, prediction, labels=labels)
        cm_percent = cm / cm.sum() * 100

        plot_data = []
        for true_index, true_label in enumerate(labels):
            for pred_index, pred_label in enumerate(labels):
                count = cm[true_index, pred_index]
                percentage = cm_percent[true_index, pred_index]
                plot_data.append({
                    "True Label": true_label,
                    "Predicted Label": pred_label,
                    "Percentage": percentage,
                    "Label": f"{int(count)}\n({percentage:.1f}%)"
                })
        plot_data = pd.DataFrame(plot_data)
        return plot_data

    def plot_roc_curve(
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
        if hasattr(model, "predict_proba"):
            # Use probability of the positive class
            y_score = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            # Use decision function score
            y_score = model.decision_function(X)
        else:
            # Use binary predictions as a last resort
            y_score = model.predict(X)

        plot_data, auc_data, auc = self._calc_plot_roc_curve(
            y_score, y, pos_label
        )
        wrapper = self._get_algo_wrapper(model.wrapper_name)
        plot = (
            pn.ggplot(plot_data, pn.aes(
                x="False Positive Rate",
                y="True Positive Rate",
                color="Type",
                linetype="Type"
            )) +
            pn.geom_line(size=1) +
            pn.geom_area(
                data=auc_data,
                fill=self.primary_color,
                alpha=0.2,
                show_legend=False
            ) +
            pn.annotate(
                "text",
                x=0.875,
                y=0.025,
                label=f"AUC = {auc:.2f}",
                color="black",
                size=12
            ) +
            pn.scale_color_manual(
                values=[self.primary_color, self.important_color],
                na_value="black"
            ) +
            pn.labs(
                title=f"ROC Curve ({wrapper.display_name})",
                color="",
                linetype=""
            ) +
            theme.brisk_theme() +
            pn.coord_fixed(ratio=1)
        )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model, is_test=X.attrs["is_test"])
        self._save_plot(output_path, metadata, plot)
        self.logger.info(
            "ROC curve with AUC = %.2f saved to %s", auc, output_path
            )

    def _calc_plot_roc_curve(
        self,
        y_score: pd.Series,
        y: np.ndarray,
        pos_label: Optional[int] = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate the plot data for the ROC curve.

        Parameters
        ----------
        y_score (pd.Series):
            The class probabilities.
        y (np.ndarray):
            The true binary labels.
        pos_label (Optional[int]):
            The label of the positive class.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, float]:
            A tuple containing the ROC curve data, the AUC data, and the AUC
            score.
        """
        fpr, tpr, _ = sk_metrics.roc_curve(y, y_score, pos_label=pos_label)
        auc = sk_metrics.roc_auc_score(y, y_score)

        roc_data = pd.DataFrame({
            "False Positive Rate": fpr,
            "True Positive Rate": tpr,
            "Type": "ROC Curve"
        })
        ref_line = pd.DataFrame({
            "False Positive Rate": [0, 1],
            "True Positive Rate": [0, 1],
            "Type": "Random Guessing"
        })
        auc_data = pd.DataFrame({
            "False Positive Rate": np.linspace(0, 1, 500),
            "True Positive Rate": np.interp(
                np.linspace(0, 1, 500), fpr, tpr
            ),
            "Type": "ROC Curve"
        })
        plot_data = pd.concat([roc_data, ref_line])
        return plot_data, auc_data, auc

    def plot_precision_recall_curve(
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

        pos_label (Optional[int]):
            The label of the positive class.
        """
        if hasattr(model, "predict_proba"):
            # Use probability of the positive class
            y_score = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            # Use decision function score
            y_score = model.decision_function(X)
        else:
            # Use binary predictions as a last resort
            y_score = model.predict(X)
        plot_data, ap_score = self._calc_plot_precision_recall_curve(
            y_score, y, pos_label
        )
        wrapper = self._get_algo_wrapper(model.wrapper_name)
        plot = (
            pn.ggplot(plot_data, pn.aes(
                x="Recall",
                y="Precision",
                color="Type",
                linetype="Type"
            )) +
            pn.geom_line(size=1) +
            pn.scale_color_manual(
                values=[self.important_color, self.primary_color],
                na_value="black"
            ) +
            pn.scale_linetype_manual(
                values=["dashed", "solid"]
            ) +
            pn.labs(
                title=f"Precision-Recall Curve ({wrapper.display_name})",
                color="",
                linetype=""
            ) +
            theme.brisk_theme() +
            pn.coord_fixed(ratio=1)
        )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model, is_test=X.attrs["is_test"])
        self._save_plot(output_path, metadata, plot)
        self.logger.info(
            "Precision-Recall curve with AP = %.2f saved to %s",
            ap_score, output_path
            )

    def _calc_plot_precision_recall_curve(
        self,
        y_score: pd.Series,
        y: np.ndarray,
        pos_label: Optional[int] = 1
    ) -> pd.DataFrame:
        """Calculate the plot data for the precision-recall curve.

        Parameters
        ----------
        y_score (pd.Series):
            The class probabilities.

        y (np.ndarray):
            The true binary labels.

        pos_label (Optional[int]):
            The label of the positive class.

        Returns
        -------
        pd.DataFrame:
            A dataframe containing the precision-recall curve data.
        """
        precision, recall, _ = sk_metrics.precision_recall_curve(
            y, y_score, pos_label=pos_label
        )
        ap_score = sk_metrics.average_precision_score(
            y, y_score, pos_label=pos_label
        )

        pr_data = pd.DataFrame({
            "Recall": recall,
            "Precision": precision,
            "Type": "PR Curve"
        })
        ap_line = pd.DataFrame({
            "Recall": [0, 1],
            "Precision": [ap_score, ap_score],
            "Type": f"AP Score = {ap_score:.2f}"
        })

        plot_data = pd.concat([pr_data, ap_line])
        return plot_data, ap_score

    # Utility Methods
    def _save_to_json(
        self,
        data: Dict[str, Any],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save dictionary to JSON file with metadata.

        Parameters
        ----------
        data : dict
            Data to save

        output_path : str
            The path to the output file.

        metadata : dict, optional
            Metadata to include, by default None
        """
        try:
            if metadata:
                data["_metadata"] = metadata

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)

        except IOError as e:
            self.logger.info(f"Failed to save JSON to {output_path}: {e}")

    def _save_plot(
        self,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        plot: Optional[pn.ggplot] = None,
        height: int = 6,
        width: int = 8
    ) -> None:
        """Save current plot to file with metadata.

        Parameters
        ----------
        output_path (str):
            The path to the output file.

        metadata (dict, optional):
            Metadata to include, by default None

        plot (ggplot, optional):
            Plotnine plot object, by default None

        height (int, optional):
            The plot height in inches, by default 6

        width (int, optional):
            The plot width in inches, by default 8
        """
        try:
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        metadata[key] = json.dumps(value)
            if plot:
                plot.save(
                    filename=output_path, format="png", metadata=metadata,
                    height=height, width=width, dpi=100
                )
            else:
                plt.savefig(output_path, format="png", metadata=metadata)
                plt.close()

        except IOError as e:
            self.logger.info(f"Failed to save plot to {output_path}: {e}")

    def save_model(self, model: base.BaseEstimator, filename: str) -> None:
        """Save model to pickle file.

        Parameters
        ----------
        model (BaseEstimator):
            The model to save.

        filename (str):
            The name for the output file (without extension).
        """
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.pkl")
        metadata = self._get_metadata(model, method_name="save_model")
        model_package = {
            "model": model,
            "metadata": metadata
        }
        joblib.dump(model_package, output_path)
        self.logger.info(
            "Saving model '%s' to '%s'.", filename, output_path
        )

    def load_model(self, filepath: str) -> base.BaseEstimator:
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
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        return joblib.load(filepath)

    def _get_metadata(
        self,
        models: Union[base.BaseEstimator, List[base.BaseEstimator]],
        method_name: Optional[str] = None,
        is_test: bool = False
    ) -> Dict[str, Any]:
        """Generate metadata for output files.

        Parameters
        ----------
        models : BaseEstimator or list of BaseEstimator
            The models to include in metadata.

        method_name (str, optional):
            The name of the calling method, by default None

        is_test (bool, optional):
            Whether the data is test data, by default False

        Returns
        -------
        dict
            Metadata including timestamp, method name, algorith wrapper name,
            and algorithm display name
        """
        metadata = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method_name if method_name else inspect.stack()[1][3],
            "models": {},
            "is_test": str(is_test)
        }
        if not isinstance(models, list):
            models = [models]

        for model in models:
            wrapper = self._get_algo_wrapper(model.wrapper_name)
            metadata["models"][wrapper.name] = wrapper.display_name

        return metadata

    def _get_algo_wrapper(
        self,
        wrapper_name: str
    ) -> algorithm_wrapper.AlgorithmWrapper:
        """Get the AlgorithmWrapper instance.

        Parameters
        ----------
        wrapper_name : str
            The name of the AlgorithmWrapper to retrieve

        Returns
        -------
        AlgorithmWrapper
            The AlgorithmWrapper instance
        """
        return self.algorithm_config[wrapper_name]

    def _get_group_index(self, is_test: bool) -> Dict[str, np.array]:
        """Get the group index for the training or test data.

        Parameters
        ----------
        is_test (bool):
            Whether the data is test data.
        """
        if self.data_has_groups:
            if is_test:
                return self.group_index_test
            return self.group_index_train
        return None

    def _get_cv_splitter(
        self,
        y: pd.Series,
        cv: int = 5,
        num_repeats: Optional[int] = None
    ) -> Tuple[model_select.BaseCrossValidator, np.array]:
        group_index = self._get_group_index(y.attrs["is_test"])

        is_categorical = False
        if y.nunique() / len(y) < 0.05:
            is_categorical = True

        if group_index:
            if is_categorical and num_repeats:
                self.logger.warning(
                    "No splitter for grouped data and repeated splitting, "
                    "using StratifiedGroupKFold instead."
                )
                splitter = model_select.StratifiedGroupKFold(n_splits=cv)
            elif not is_categorical and num_repeats:
                self.logger.warning(
                    "No splitter for grouped data and repeated splitting, "
                    "using GroupKFold instead."
                )
                splitter = model_select.GroupKFold(n_splits=cv)
            elif is_categorical:
                splitter = model_select.StratifiedGroupKFold(n_splits=cv)
            else:
                splitter = model_select.GroupKFold(n_splits=cv)

        else:
            if is_categorical and num_repeats:
                splitter = model_select.RepeatedStratifiedKFold(n_splits=cv)
            elif not is_categorical and num_repeats:
                splitter = model_select.RepeatedKFold(n_splits=cv)
            elif is_categorical:
                splitter = model_select.StratifiedKFold(n_splits=cv)
            else:
                splitter = model_select.KFold(n_splits=cv)

        if group_index:
            indices = group_index["indices"]
        else:
            indices = None

        return splitter, indices
