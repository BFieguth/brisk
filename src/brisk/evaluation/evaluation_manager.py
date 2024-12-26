"""Provides the EvaluationManager class for model evaluation and visualization.

Exports:
    - EvaluationManager: A class that provides methods for evaluating models, 
        generating plots, and comparing models. These methods are used when 
        building a training workflow.
"""

import copy
import datetime
import inspect
import itertools
import json
import logging
import os
from typing import Dict, List, Optional, Any, Union

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotnine as pn

from sklearn import base
from sklearn import ensemble
from sklearn import inspection
import sklearn.model_selection as model_select
import sklearn.metrics as sk_metrics
from sklearn import tree

from brisk.utility import algorithm_wrapper
from brisk.theme import theme

matplotlib.use("Agg")


class EvaluationManager:
    """A class for evaluating machine learning models and plotting results.

    This class provides methods for model evaluation, including calculating 
    metrics, generating plots, comparing models, and hyperparameter tuning. It 
    is designed to be used within a training workflow.

    Attributes:
        algorithm_config (dict): Configuration for algorithms.
        metric_config (object): Configuration for evaluation metrics.
    """
    def __init__(
        self,
        algorithm_config: List[algorithm_wrapper.AlgorithmWrapper],
        metric_config: Any,
        output_dir: str,
        split_metadata: Dict[str, Any],
        logger: Optional[logging.Logger]=None,
    ):
        """
        Initialize the EvaluationManager with algorithm and scoring
        configurations.

        Args:
            algorithm_config (Dict[str, Any]): Configuration for algorithms.
            metric_config (Any): Configuration for evaluation metrics.
            output_dir (str): Directory to save results.
            split_metadata (Dict[str, Any]): Metadata to include in metric 
            calculations.
            logger (Optional[logging.Logger]): Logger instance to use.
        """
        self.algorithm_config = algorithm_config
        self.metric_config = copy.deepcopy(metric_config)
        self.metric_config.set_split_metadata(split_metadata)
        self.output_dir = output_dir
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
        """
        Evaluate the given model on the provided metrics and save the results.

        Args:
            model (BaseEstimator): The trained machine learning model to 
            evaluate.
            X (pd.DataFrame): The feature data to use for evaluation.
            y (pd.Series): The target data to use for evaluation.
            metrics (List[str]): A list of metric names to calculate.
            filename (str): The name of the output file without extension.

        Returns:
            None
        """
        predictions = model.predict(X)
        results = {}

        for metric_name in metrics:
            display_name = self.metric_config.get_name(metric_name)
            scorer = self.metric_config.get_metric(metric_name)
            if scorer is not None:
                score = scorer(y, predictions)
                results[display_name] = score
            else:
                self.logger.info(f"Scorer for {metric_name} not found.")

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        metadata = self._get_metadata(model)
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

    def evaluate_model_cv(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
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
        results = {}

        for metric_name in metrics:
            display_name = self.metric_config.get_name(metric_name)
            scorer = self.metric_config.get_scorer(metric_name)
            if scorer is not None:
                scores = model_select.cross_val_score(
                    model, X, y, scoring=scorer, cv=cv
                    )
                results[display_name] = {
                    "mean_score": scores.mean(),
                    "std_dev": scores.std(),
                    "all_scores": scores.tolist()
                }
            else:
                self.logger.info(f"Scorer for {metric_name} not found.")

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        metadata = self._get_metadata(model)
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

    def compare_models(
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
            Dict[str, Dict[str, float]]: A dictionary containing the metric 
            results for each model.
        """
        comparison_results = {}

        if not models:
            raise ValueError("At least one model must be provided")

        model_names = [model.__class__.__name__ for model in models]

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

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        metadata = self._get_metadata(models=models)
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
        return comparison_results

    def plot_pred_vs_obs(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
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
        prediction = model.predict(X)

        plot_data = pd.DataFrame({
            "Observed": y_true,
            "Predicted": prediction
        })
        max_range = plot_data[['Observed', 'Predicted']].max().max()
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
                title="Predicted vs. Observed Values"
            ) +
            pn.coord_fixed(
                xlim=[0, max_range],
                ylim=[0, max_range]
            ) +
            theme.brisk_theme()
        )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model)
        self._save_plot(output_path, metadata, plot=plot)
        self.logger.info(
            "Predicted vs. Observed plot saved to '%s'.", output_path
        )

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
        """
        Plot a learning curve for the given model and save the plot.

        Args:
            model (BaseEstimator): The machine learning model to evaluate.

            X_train (pd.DataFrame): The input features of the training set.
            
            y_train (pd.Series): The target values of the training set.
            
            cv (int): Number of cross-validation folds. Defaults to 5.
            
            num_repeats (int): Number of times to repeat the cross-validation. 
            Defaults to 1.
            
            metric (str): The scoring metric to use. Defaults to 
            "neg_mean_absolute_error".
            
            filename (str): The name of the output PNG file (without extension).

        Returns:
            None
        """
        method_name = model.__class__.__name__

        cv = model_select.RepeatedKFold(n_splits=cv, n_repeats=num_repeats)

        scorer = self.metric_config.get_scorer(metric)

        # Generate learning curve data
        train_sizes, train_scores, test_scores, fit_times, _ = (
            model_select.learning_curve(
                model, X_train, y_train, cv=cv, n_jobs=n_jobs,
                train_sizes=np.linspace(0.1, 1.0, 5), return_times=True,
                scoring=scorer
            )
        )

        # Calculate means and standard deviations
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Create subplots
        _, axes = plt.subplots(1, 3, figsize=(16, 6))
        plt.rcParams.update({"font.size": 12})

        # Plot Learning Curve
        display_name = self.metric_config.get_name(metric)
        axes[0].set_title(f"Learning Curve ({method_name})", fontsize=20)
        axes[0].set_xlabel("Training Examples", fontsize=12)
        axes[0].set_ylabel(display_name, fontsize=12)
        axes[0].grid()
        axes[0].fill_between(
            train_sizes, train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, alpha=0.1, color="r"
            )
        axes[0].fill_between(
            train_sizes, test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, alpha=0.1, color="g"
            )
        axes[0].plot(
            train_sizes, train_scores_mean, "o-", color="r",
            label="Training Score"
            )
        axes[0].plot(
            train_sizes, test_scores_mean, "o-", color="g",
            label="Cross-Validation Score"
            )
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, "o-")
        axes[1].fill_between(
            train_sizes, fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std, alpha=0.1
            )
        axes[1].set_xlabel("Training Examples", fontsize=12)
        axes[1].set_ylabel("Fit Times", fontsize=12)
        axes[1].set_title("Scalability of the Model", fontsize=16)

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, "o-")
        axes[2].fill_between(
            fit_times_mean, test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, alpha=0.1
            )
        axes[2].set_xlabel("Fit Times", fontsize=12)
        axes[2].set_ylabel(display_name, fontsize=12)
        axes[2].set_title("Performance of the Model", fontsize=16)

        plt.tight_layout()

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model)
        self._save_plot(output_path, metadata)
        self.logger.info(f"Learning Curve plot saved to '{output_path}''.")

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

        Args:
            model (BaseEstimator): The machine learning model to evaluate.

            X (pd.DataFrame): The feature data.

            y (pd.Series): The target data.

            threshold (Union[int, float]): The number of features or the 
            threshold to filter features by importance.

            feature_names (List[str]): A list of feature names corresponding to 
                the columns in X.

            filename (str): The name of the output PNG file (without extension).

            metric (str): The metric to use for evaluation.

            num_rep (int): The number of repetitions for calculating importance.

        Returns:
            None
        """
        scorer = self.metric_config.get_scorer(metric)
        display_name = self.metric_config.get_name(metric)

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
            above_threshold = importance >= threshold
            importance = importance[above_threshold]
            feature_names = [
                feature_names[i] for i in range(len(feature_names))
                if above_threshold[i]
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
        importance_data['Feature'] = pd.Categorical(
            importance_data['Feature'], 
            categories=importance_data.sort_values('Importance')['Feature'], 
            ordered=True
        )
        plot = (
            pn.ggplot(importance_data, pn.aes(x="Feature", y="Importance")) +
            pn.geom_bar(stat="identity", fill=self.primary_color) +
            pn.coord_flip() +
            pn.labs(
                x="Feature", y=f"Importance ({display_name})",
                title="Feature Importance"
            ) +
            theme.brisk_theme()
        )
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model)
        self._save_plot(output_path, metadata, plot, plot_height, plot_width)
        self.logger.info(
            "Feature Importance plot saved to '%s'.", output_path
        )

    def plot_residuals(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        filename: str,
        add_fit_line: bool = False
    ) -> None:
        """Plot the residuals of the model and save the plot.

        Args:
            model (BaseEstimator): The trained machine learning model.
            
            X (pd.DataFrame): The feature data.
            
            y (pd.Series): The true target values.
            
            filename (str): The name of the output PNG file (without extension).

            add_fit_line (bool): Whether to add a line of best fit to the plot.

        Returns:
            None
        """
        predictions = model.predict(X)
        residuals = y - predictions

        plot_data = pd.DataFrame({
            "Observed": y,
            "Residual": residuals
        })
        plot = (
            pn.ggplot(plot_data, pn.aes(x="Observed", y="Residual")) +
            pn.geom_point(
                color="black", size=3, stroke=0.25, fill=self.primary_color
            ) +
            pn.geom_abline(
                slope=0, intercept=0, color=self.important_color,
                linetype="dashed", size=1.5
            ) +
            pn.ggtitle("Residuals (Observed - Predicted)") +
            theme.brisk_theme()
        )

        if add_fit_line:
            fit = np.polyfit(plot_data['Observed'], plot_data['Residual'], 1)
            fit_line = np.polyval(fit, plot_data['Observed'])
            plot += (
                pn.geom_line(
                    pn.aes(x='Observed', y=fit_line, group=1), 
                    color=self.accent_color, size=1
                )
            )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model)
        self._save_plot(output_path, metadata, plot)
        self.logger.info(
            "Residuals plot saved to '%s'.", output_path
        )

    def plot_model_comparison(
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
        model_names = [model.__class__.__name__ for model in models]
        metric_values = []

        scorer = self.metric_config.get_metric(metric)
        display_name = self.metric_config.get_name(metric)

        for model in models:
            predictions = model.predict(X)
            if scorer is not None:
                score = scorer(y, predictions)
                metric_values.append(round(score, 3))
            else:
                self.logger.info(f"Scorer for {metric} not found.")
                return

        plot_data = pd.DataFrame({
            "Model": model_names,
            "Score": metric_values,
        })
        title = f"Model Comparison on {display_name}"
        plot = (
            pn.ggplot(plot_data, pn.aes(x="Model", y="Score")) +
            pn.geom_bar(stat="identity", fill=self.primary_color) +
            pn.geom_text(
                pn.aes(label="Score"), position=pn.position_stack(vjust=0.5),
                color="white", size=16
            ) +
            pn.ggtitle(title=title) +
            pn.ylab(display_name) +
            theme.brisk_theme()
        )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(models)
        self._save_plot(output_path, metadata, plot)
        self.logger.info(
            "Model Comparison plot saved to '%s'.", output_path
        )
        plt.close()

    def hyperparameter_tuning(
        self,
        model: base.BaseEstimator,
        method: str,
        algorithm_name: str,
        X_train: pd.DataFrame, # pylint: disable=C0103
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

            method (str): The search method to use ("grid" or "random").

            algorithm_name (str): The name of the algorithm for which the 
            hyperparameter grid is being used.

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
        param_grid = next(
            algo.get_hyperparam_grid() for algo in self.algorithm_config
            if algo.name == algorithm_name
            )
        display_name = next(
            algo.display_name for algo in self.algorithm_config 
            if algo.name == algorithm_name
        )

        cv = model_select.RepeatedKFold(n_splits=kf, n_repeats=num_rep)
        search = searcher(
            estimator=model, param_grid=param_grid, n_jobs=n_jobs, cv=cv,
            scoring=score
        )
        search_result = search.fit(X_train, y_train)
        tuned_model = next(
            algo.instantiate_tuned(search_result.best_params_)
            for algo in self.algorithm_config if algo.name == algorithm_name
            )
        tuned_model.fit(X_train, y_train)
        self.logger.info(
            "Hyperparameter optimization for %s complete.", 
            model.__class__.__name__
            )

        if plot_results:
            metadata = self._get_metadata(model)
            self._plot_hyperparameter_performance(
                param_grid, search_result, algorithm_name, metadata,
                display_name
            )
        return tuned_model

    def _plot_hyperparameter_performance(
        self,
        param_grid: Dict[str, Any],
        search_result: Any,
        algorithm_name: str,
        metadata: Dict[str, Any],
        display_name: str
    ) -> None:
        """Plot the performance of hyperparameter tuning.

        Args:
            param_grid (Dict[str, Any]): The hyperparameter grid used for 
            tuning.

            search_result (Any): The result from cross-validation during tuning.
            
            algorithm_name (str): The name of the algorithm.
            
            metadata (Dict[str, Any]): Metadata to be included with the plot.

            display_name (str): The name of the algorithm to use in the labels.

        Returns:
            None
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
        """Plot the performance of a single hyperparameter.

        Args:
            param_values (List[Any]): The values of the hyperparameter.
            mean_test_score (List[float]): The mean test scores for each 
                hyperparameter value.
            param_name (str): The name of the hyperparameter.
            algorithm_name (str): The name of the algorithm.
            metadata (Dict[str, Any]): Metadata to be included with the plot.
            display_name (str): The name of the algorithm to use in the labels.

        Returns:
            None
        """
        param_name = param_name.capitalize()
        title = f"Hyperparameter Performance: {display_name}"
        plot_data = pd.DataFrame({
            "Hyperparameter": param_values,
            "Mean Test Score": mean_test_score,
        })
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
        """Plot the performance of two hyperparameters in 3D.

        Args:
            param_grid (Dict[str, List[Any]]): The hyperparameter grid used for 
            tuning.
            
            search_result (Any): The result from cross-validation during tuning.
            
            param_names (List[str]): The names of the two hyperparameters.
            
            algorithm_name (str): The name of the algorithm.
            
            metadata (Dict[str, Any]): Metadata to be included with the plot.

            display_name (str): The name of the algorithm to use in the labels.

        Returns:
            None
        """
        mean_test_score = search_result.cv_results_["mean_test_score"].reshape(
            len(param_grid[param_names[0]]),
            len(param_grid[param_names[1]])
        )
        # Create meshgrid for parameters
        X, Y = np.meshgrid( # pylint: disable=C0103
            param_grid[param_names[0]], param_grid[param_names[1]]
            )

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, mean_test_score.T, cmap="viridis")
        ax.set_xlabel(param_names[0], fontsize=12)
        ax.set_ylabel(param_names[1], fontsize=12)
        ax.set_zlabel("Mean Test Score", fontsize=12)
        ax.set_title(
            f"Hyperparameter Performance: {algorithm_name}", fontsize=16
        )
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.output_dir, f"{algorithm_name}_hyperparam_3Dplot.png"
            )
        self._save_plot(output_path, metadata)
        self.logger.info(
            "Hyperparameter performance plot saved to '%s'.", output_path
            )

    def confusion_matrix(
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
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
        y_pred = model.predict(X)
        labels = np.unique(y).tolist()
        cm = sk_metrics.confusion_matrix(y, y_pred, labels=labels).tolist()
        data = {
            "confusion_matrix": cm,
            "labels": labels
            }

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        metadata = self._get_metadata(model)
        self._save_to_json(data, output_path, metadata)

        header = " " * 10 + " ".join(f"{label:>10}" for label in labels) + "\n"
        rows = [f"{label:>10} " + " ".join(f"{count:>10}" for count in row)
                for label, row in zip(labels, cm)]
        table = header + "\n".join(rows)
        confusion_log = f"Confusion Matrix:\n{table}"
        self.logger.info(confusion_log)

    def plot_confusion_heatmap(
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str
    ) -> None:
        """
        Generate a heatmap of the confusion matrix for a model and dataset.

        Args:
            model (Any): Trained classification model with a `predict` method.
            
            X (np.ndarray): Input features.
            
            y (np.ndarray): Target labels.
            
            filename (str): Path to save the confusion matrix heatmap image.
            
            labels (Optional[list]): List of class labels for display on the 
            heatmap axes.

        Returns:
            None
        """
        y_pred = model.predict(X)
        labels = np.unique(y).tolist()
        cm = sk_metrics.confusion_matrix(y, y_pred, labels=labels)
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

        plot = (
            pn.ggplot(plot_data, pn.aes(
                x='Predicted Label',
                y='True Label',
                fill='Percentage'
            )) +
            pn.geom_tile() +
            pn.geom_text(pn.aes(label='Label'), color='black') +
            pn.scale_fill_gradient(
                low="white",
                high=self.primary_color,
                name='Percentage (%)',
                limits=(0, 100)
            ) +
            pn.ggtitle('Confusion Matrix Heatmap') +
            theme.brisk_theme()
        )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model)
        self._save_plot(output_path, metadata, plot)
        self.logger.info(f"Confusion matrix heatmap saved to {output_path}")

    def plot_roc_curve(
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str
    ) -> None:
        """
        Generate a ROC curve with AUC for a binary classification model and 
        dataset.

        Args:
            model (Any): Trained binary classification model with a 
            `predict_proba` method.

            X (np.ndarray): Input features.
            
            y (np.ndarray): True binary labels.
            
            filename (str): Path to save the ROC curve image.

        Returns:
            None
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

        fpr, tpr, _ = sk_metrics.roc_curve(y, y_score)
        auc = sk_metrics.roc_auc_score(y, y_score)

        roc_data = pd.DataFrame({
            "False Positive Rate": fpr,
            "True Positive Rate": tpr,
            "Type": "ROC Curve"
        })
        ref_line = pd.DataFrame({
            'False Positive Rate': [0, 1],
            'True Positive Rate': [0, 1],
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
                values=[self.primary_color, self.important_color]
            ) +
            pn.labs(
                title=f"ROC Curve for {model.__class__.__name__}",
                color='',
                linetype=''
            ) +
            theme.brisk_theme() +
            pn.coord_fixed(ratio=1)
        )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model)
        self._save_plot(output_path, metadata, plot)
        self.logger.info(
            "ROC curve with AUC = %.2f saved to %s", auc, output_path
            )

    def plot_precision_recall_curve(
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str
    ) -> None:
        """
        Generate and save a Precision-Recall (PR) curve with Average Precision 
        (AP) for a given binary classification model and dataset.

        Args:
            model (Any): Trained binary classification model.
            X (np.ndarray): Input features.
            y (np.ndarray): True binary labels.
            filename (str): Path to save the PR curve image.

        Returns:
            None
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

        precision, recall, _ = sk_metrics.precision_recall_curve(y, y_score)
        ap_score = sk_metrics.average_precision_score(y, y_score)

        plt.figure(figsize=(8, 6))
        plt.plot(
            recall, precision, label=f"PR Curve (AP = {ap_score:.2f})",
            color="purple"
            )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model)
        self._save_plot(output_path, metadata)
        self.logger.info(
            "Precision-Recall curve with AP = %.2f saved to %s",
            ap_score, output_path
            )

    # Utility Methods
    def _save_to_json(
        self,
        data: Dict[str, Any],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save a dictionary to a JSON file, including metadata.

        Args:
            data (Dict[str, Any]): The data to save.
            output_path (str): The path to the output file.
            metadata (Optional[Dict[str, Any]]): Metadata to be included with 
                the data. Defaults to None.

        Returns:
            None
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
        height = 6,
        width = 8
    ) -> None:
        """Save the current matplotlib plot to a PNG file, including metadata.

        Args:
            output_path (str): The full path (including filename) where the plot 
                will be saved.
            metadata (Optional[Dict[str, Any]]): Metadata to be included with 
                the plot. Defaults to None.

        Returns:
            None
        """
        try:
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
        """Save the model to a file in pickle format.

        Args:
            model (BaseEstimator): The model to save.
            filename (str): The name of the output file (without extension).

        Returns:
            None
        """
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.pkl")
        joblib.dump(model, output_path)
        self.logger.info(
            "Saving model '%s' to '%s'.", filename, output_path
            )

    def load_model(self, filepath: str) -> base.BaseEstimator:
        """Load a model from a pickle file.

        Args:
            filepath (str): The path to the file containing the saved model.

        Returns:
            BaseEstimator: The loaded model.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        return joblib.load(filepath)

    def _get_metadata(
        self,
        models: Union[base.BaseEstimator, List[base.BaseEstimator]],
        method_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate metadata for saving output files (JSON, PNG, etc.).

        Args:
            models (Union[base.BaseEstimator, List[base.BaseEstimator]]): 
                A single model or a list of models to include in metadata.

            method_name (Optional[str]): The name of the calling method.

        Returns:
            Dict[str, Any]: A dictionary containing metadata such as method 
            name, timestamp, and model names.
        """
        metadata = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method_name if method_name else inspect.stack()[1][3]
        }

        if isinstance(models, tuple):
            metadata["models"] = [model.__class__.__name__ for model in models]
        else:
            metadata["models"] = [models.__class__.__name__]

        metadata = {
            k: str(v) if not isinstance(v, str)
            else v for k, v in metadata.items()
            }
        return metadata
