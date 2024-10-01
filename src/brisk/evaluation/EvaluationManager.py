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
import os
from typing import Dict, List, Optional, Any, Union

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection as model_select
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.inspection as inspection
import sklearn.base as base
matplotlib.use('Agg')

class EvaluationManager:
    """A class for evaluating machine learning models and generating visualizations.

    This class provides methods for model evaluation, including calculating 
    metrics, generating plots, comparing models, and hyperparameter tuning. It 
    is designed to be used within a training workflow.

    Attributes:
        method_config (dict): Configuration for model methods.
        metric_config (object): Configuration for evaluation metrics.
    """
    def __init__(self, method_config: Dict[str, Any], metric_config: Any):
        """
        Initializes the EvaluationManager with method and scoring configurations.

        Args:
            method_config (Dict[str, Any]): Configuration for model methods.
            metric_config (Any): Configuration for evaluation metrics.
        """
        self.method_config = method_config
        self.metric_config = metric_config

    # Evaluation Tools
    def evaluate_model(
        self, 
        model: base.BaseEstimator, 
        X: pd.DataFrame, 
        y: pd.Series, 
        metrics: List[str], 
        filename: str
    ) -> None:
        """ Evaluate the given model on the provided metrics and save the results.

        Args:
            model (BaseEstimator): The trained machine learning model to evaluate.
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
            scorer = self.metric_config.get_metric(metric_name)
            if scorer is not None:
                score = scorer(y, predictions)
                results[metric_name] = score
                # print(f"{metric_name}: {score}")
            else:
                print(f"Scorer for {metric_name} not found.")
        
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        metadata = self._get_metadata(model)
        self._save_to_json(results, output_path, metadata)

    def evaluate_model_cv(
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
        results = {}

        for metric_name in metrics:
            scorer = self.metric_config.get_scorer(metric_name)
            if scorer is not None:
                scores = model_select.cross_val_score(
                    model, X, y, scoring=scorer, cv=cv
                    )
                results[metric_name] = {
                    "mean_score": scores.mean(),
                    "std_dev": scores.std(),
                    "all_scores": scores.tolist()
                }
                # print(
                #     f"{metric_name} - Mean: {scores.mean()}, "
                #     f"Std Dev: {scores.std()}"
                #     )
            else:
                print(f"Scorer for {metric_name} not found.")

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        metadata = self._get_metadata(model)
        self._save_to_json(results, output_path, metadata)

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
            Dict[str, Dict[str, float]]: A dictionary containing the metric results 
                for each model.
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
                if scorer is not None:
                    score = scorer(y, predictions)
                    results[metric_name] = score
                else:
                    print(f"Scorer for {metric_name} not found.")
            
            comparison_results[model_name] = results

        # Calculate the difference between models for each metric
        if calculate_diff and len(models) > 1:
            comparison_results["differences"] = {}
            model_pairs = list(itertools.combinations(model_names, 2))
                    
            for metric_name in metrics:
                comparison_results["differences"][metric_name] = {}

                for model_a, model_b in model_pairs:
                    score_a = comparison_results[model_a][metric_name]
                    score_b = comparison_results[model_b][metric_name]
                    diff = score_b - score_a
                    comparison_results["differences"][metric_name][f"{model_b} - {model_a}"] = diff

        output_path = os.path.join(self.output_dir, f"{filename}.json")
        metadata = self._get_metadata(models=models)
        self._save_to_json(comparison_results, output_path, metadata)
        
        return comparison_results

    def plot_pred_vs_obs(
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
        prediction = model.predict(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, prediction, edgecolors=(0, 0, 0))
        plt.plot(
            [min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2
            )
        plt.xlabel("Observed Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs. Observed Values")
        plt.grid(True)

        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model)
        self._save_plot(output_path, metadata)      

    def plot_learning_curve(
        self, 
        model: base.BaseEstimator, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        cv: int = 5, 
        num_repeats: int = 1, 
        scoring: str = 'neg_mean_absolute_error', 
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
            scoring (str): The scoring metric to use. Defaults to 'neg_mean_absolute_error'.
            filename (str): The name of the output PNG file (without extension).

        Returns:
            None
        """
        method_name = model.__class__.__name__

        cv = model_select.RepeatedKFold(n_splits=cv, n_repeats=num_repeats)

        # Generate learning curve data
        train_sizes, train_scores, test_scores, fit_times, _ = model_select.learning_curve(
            model, X_train, y_train, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5), return_times=True, 
            scoring=scoring
        )

        # Calculate means and standard deviations
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        plt.rcParams.update({'font.size': 12})

        # Plot Learning Curve
        axes[0].set_title(f"Learning Curve ({method_name})", fontsize=20)
        axes[0].set_xlabel("Training Examples", fontsize=12)
        axes[0].set_ylabel(scoring, fontsize=12) 
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
            train_sizes, train_scores_mean, 'o-', color="r", 
            label="Training Score"
            )
        axes[0].plot(
            train_sizes, test_scores_mean, 'o-', color="g", 
            label="Cross-Validation Score"
            )
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(
            train_sizes, fit_times_mean - fit_times_std, 
            fit_times_mean + fit_times_std, alpha=0.1
            )
        axes[1].set_xlabel("Training Examples", fontsize=12)
        axes[1].set_ylabel("Fit Times", fontsize=12)
        axes[1].set_title("Scalability of the Model", fontsize=16)

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(
            fit_times_mean, test_scores_mean - test_scores_std, 
            test_scores_mean + test_scores_std, alpha=0.1
            )
        axes[2].set_xlabel("Fit Times", fontsize=12)
        axes[2].set_ylabel(scoring, fontsize=12)
        axes[2].set_title("Performance of the Model", fontsize=16)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model)
        self._save_plot(output_path, metadata)

    def plot_feature_importance(
        self, 
        model: base.BaseEstimator, 
        X: pd.DataFrame, 
        y: pd.Series, 
        filter: Union[int, float], 
        feature_names: List[str], 
        filename: str, 
        scoring: str, 
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
            scoring (str): The scoring metric to use for evaluation.
            num_rep (int): The number of repetitions for calculating importance.

        Returns:
            None
        """
        metric = self.metric_config.get_scorer(scoring)

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
                model, X=X, y=y, scoring=metric, n_repeats=num_rep
                )
            importance = results.importances_mean

        if isinstance(filter, int):
            sorted_indices = np.argsort(importance)[::-1]
            importance = importance[sorted_indices[:filter]]
            feature_names = [feature_names[i] for i in sorted_indices[:filter]]
        elif isinstance(filter, float):
            above_threshold = importance >= filter
            importance = importance[above_threshold]
            feature_names = [
                feature_names[i] for i in range(len(feature_names)) 
                if above_threshold[i]
                ]

        plt.barh(feature_names, importance)
        plt.xticks(rotation=90)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance', fontsize=16)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{filename}.png')
        metadata = self._get_metadata(model)
        self._save_plot(output_path, metadata)

    def plot_residuals(
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
        predictions = model.predict(X)
        residuals = y - predictions

        plt.figure(figsize=(8, 6))
        plt.scatter(y, residuals, label='Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Observed', fontsize=12)
        plt.ylabel('Residual', fontsize=12)
        plt.title('Residual Plot', fontsize=16)
        plt.legend()
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(model)
        self._save_plot(output_path, metadata)

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
        
        for idx, model in enumerate(models):
            predictions = model.predict(X)
            scorer = self.metric_config.get_metric(metric)
            if scorer is not None:
                score = scorer(y, predictions)
                metric_values.append(score)
                # print(f"{model_names[idx]} - {metric}: {score}")
            else:
                print(f"Scorer for {metric} not found.")
                return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, metric_values, color='lightblue')
        # Add labels to each bar
        for bar, value in zip(bars, metric_values):
            plt.text(
                bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{value:.3f}', ha='center', va='bottom'
                )
        plt.xlabel("Models", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f"Model Comparison on {metric}", fontsize=16)
        
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        metadata = self._get_metadata(models)
        self._save_plot(output_path, metadata)
        plt.close()

    def hyperparameter_tuning(
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
        if method == "grid":
            searcher = model_select.GridSearchCV
        elif method == "random":
            searcher = model_select.RandomizedSearchCV
        else:
            raise ValueError(
                f"method must be one of (grid, random). {method} was entered."
                )
        
        score = self.metric_config.get_scorer(scorer)
        param_grid = self.method_config[method_name].get_hyperparam_grid()

        cv = model_select.RepeatedKFold(n_splits=kf, n_repeats=num_rep)
        search = searcher(
            estimator=model, param_grid=param_grid, n_jobs=n_jobs, cv=cv,
            scoring=score
        )
        search_result = search.fit(X_train, y_train)
        tuned_model = self.method_config[method_name].instantiate_tuned(
            search_result.best_params_
            )
        tuned_model.fit(X_train, y_train)

        if plot_results:
            metadata = self._get_metadata(model)
            self._plot_hyperparameter_performance(
                param_grid, search_result, method_name, metadata
            )
        return tuned_model

    def _plot_hyperparameter_performance(
        self, 
        param_grid: Dict[str, Any], 
        search_result: Any, 
        method_name: str, 
        metadata: Dict[str, Any]
    ) -> None:
        """Plot the performance of hyperparameter tuning.

        Args:
            param_grid (Dict[str, Any]): The hyperparameter grid used for tuning.
            search_result (Any): The results from cross-validation during tuning.
            method_name (str): The name of the model method.
            metadata (Dict[str, Any]): Metadata to be included with the plot.

        Returns:
            None
        """ 
        param_keys = list(param_grid.keys())
        # print(
        #     f"Hyperparam plot for {method_name} has {len(param_keys)} hyperparams"
        #     )

        if len(param_keys) == 0:
            return

        elif len(param_keys) == 1:
            self._plot_1d_performance(
                param_values=param_grid[param_keys[0]], 
                mean_test_score=search_result.cv_results_['mean_test_score'], 
                param_name=param_keys[0], 
                method_name=method_name,
                metadata=metadata
            )
        elif len(param_keys) == 2:
            self._plot_3d_surface(
                param_grid=param_grid, 
                search_result=search_result, 
                param_names=param_keys, 
                method_name=method_name,
                metadata=metadata
            )
        else:
            print("Higher dimensional visualization not implemented yet")

    def _plot_1d_performance(
        self, 
        param_values: List[Any], 
        mean_test_score: List[float], 
        param_name: str, 
        method_name: str, 
        metadata: Dict[str, Any]
    ) -> None:
        """Plot the performance of a single hyperparameter.

        Args:
            param_values (List[Any]): The values of the hyperparameter.
            mean_test_score (List[float]): The mean test scores for each 
                hyperparameter value.
            param_name (str): The name of the hyperparameter.
            method_name (str): The name of the model method.
            metadata (Dict[str, Any]): Metadata to be included with the plot.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(
            param_values, mean_test_score, marker='o', linestyle='-', color='b'
            )
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel("Mean Test Score", fontsize=12)
        plt.title(
            f"Hyperparameter Performance: {method_name} ({param_name})", 
            fontsize=16
            )
        
        for i, score in enumerate(mean_test_score):
            plt.text(
                param_values[i], score, f'{score:.2f}', ha='center', va='bottom'
                )
        
        plt.grid(True)
        plt.tight_layout()
        output_path = os.path.join(
            self.output_dir, f"{method_name}_hyperparam_{param_name}.png"
            )
        self._save_plot(output_path, metadata)

    def _plot_3d_surface(
        self, 
        param_grid: Dict[str, List[Any]], 
        search_result: Any, 
        param_names: List[str], 
        method_name: str, 
        metadata: Dict[str, Any]
    ) -> None:
        """Plot the performance of two hyperparameters in 3D.

        Args:
            param_grid (Dict[str, List[Any]]): The hyperparameter grid used for tuning.
            search_result (Any): The results from cross-validation during tuning.
            param_names (List[str]): The names of the two hyperparameters.
            method_name (str): The name of the model method.
            metadata (Dict[str, Any]): Metadata to be included with the plot.

        Returns:
            None
        """
        mean_test_score = search_result.cv_results_['mean_test_score'].reshape(
            len(param_grid[param_names[0]]), 
            len(param_grid[param_names[1]])
        )
        # Create meshgrid for parameters
        X, Y = np.meshgrid(
            param_grid[param_names[0]], param_grid[param_names[1]]
            )

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, mean_test_score.T, cmap='viridis')
        ax.set_xlabel(param_names[0], fontsize=12)
        ax.set_ylabel(param_names[1], fontsize=12)
        ax.set_zlabel('Mean Test Score', fontsize=12)
        ax.set_title(f"Hyperparameter Performance: {method_name}", fontsize=16)
        output_path = os.path.join(
            self.output_dir, f"{method_name}_hyperparam_3Dplot.png"
            )
        self._save_plot(output_path, metadata)       

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

            with open(output_path, "w") as file:
                json.dump(data, file, indent=4)

        except IOError as e:
            print(f"Failed to save JSON to {output_path}: {e}")

    def _save_plot(
        self, 
        output_path: str, 
        metadata: Optional[Dict[str, Any]] = None
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
            plt.savefig(output_path, format="png", metadata=metadata)
            plt.close()

        except IOError as e:
            print(f"Failed to save plot to {output_path}: {e}")

    def with_config(self, **kwargs: Any) -> 'EvaluationManager':
        """Create a copy of the current evaluator with additional configuration.

        Args:
            kwargs: Key-value pairs of attributes to add/modify in the copy.

        Returns:
            EvaluationManager: A copy of the current evaluator with updated attributes.
        """
        eval_copy = copy.copy(self)
        for key, value in kwargs.items():
            setattr(eval_copy, key, value)
        return eval_copy

    def save_model(self, model: base.BaseEstimator, filename: str) -> None:
        """Save the model to a file in pickle format.

        Args:
            model (BaseEstimator): The model to save.
            filename (str): The name of the output file (without extension).

        Returns:
            None
        """
        output_path = os.path.join(self.output_dir, f"{filename}.pkl")
        joblib.dump(model, output_path)

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
            method_name (Optional[str]): The name of the calling method (if available).

        Returns:
            Dict[str, Any]: A dictionary containing metadata such as method name, 
                timestamp, and model names.
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
    