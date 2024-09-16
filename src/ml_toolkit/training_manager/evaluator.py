import copy
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as model_select
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.inspection as inspection
import sklearn.base as base

# TODO
# 7. Perform Hyperparameter Optimization
# 8. Save model
# 9. Load model
# 10. Compare Models (on specified metric w/ same data)
# 11. Plot model comparison (plot the metric for each model)
# 12. Plot hyperparameter performance

class Evaluator:
    def __init__(self, method_config, scoring_config):
        self.method_config = method_config
        self.scoring_config = scoring_config

    # Evaluation Tools
    def evaluate_model(
        self, 
        model, 
        X, 
        y, 
        metrics: List[str], 
        filename: str
    ) -> None:
        """
        Calculate and save the scores for the given model and metrics.

        Args:
            model: The trained machine learning model to evaluate.
            X: The feature data to use for evaluation.
            y: The target data to use for evaluation.
            metrics (list): A list of metric names to calculate.
            output_dir (str): The directory to save the results.
            filename (str): The name of the output file without extension.
        """
        predictions = model.predict(X)
        results = {}

        for metric_name in metrics:
            scorer = self.scoring_config.get_metric(metric_name)
            if scorer is not None:
                score = scorer(y, predictions)
                results[metric_name] = score
                print(f"{metric_name}: {score}")
            else:
                print(f"Scorer for {metric_name} not found.")
        
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        self.__save_to_json(results, output_path)

    def evaluate_model_cv(
        self, 
        model, 
        X, 
        y, 
        metrics: List[str], 
        filename: str, 
        cv: int = 5
    ) -> None:
        """
        Evaluate the model using cross-validation and save the scores for the given metrics.

        Args:
            model: The machine learning model to evaluate.
            X: The feature data to use for evaluation.
            y: The target data to use for evaluation.
            metrics (list): A list of metric names to calculate.
            output_dir (str): The directory to save the results.
            filename (str): The name of the output file without extension.
            cv (int): The number of cross-validation folds. Default is 5.
        """
        results = {}

        for metric_name in metrics:
            scorer = self.scoring_config.get_scorer(metric_name)
            if scorer is not None:
                scores = model_select.cross_val_score(
                    model, X, y, scoring=scorer, cv=cv
                    )
                results[metric_name] = {
                    "mean_score": scores.mean(),
                    "std_dev": scores.std(),
                    "all_scores": scores.tolist()
                }
                print(
                    f"{metric_name} - Mean: {scores.mean()}, "
                    f"Std Dev: {scores.std()}"
                    )
            else:
                print(f"Scorer for {metric_name} not found.")

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        self.__save_to_json(results, output_path)

    def plot_pred_vs_obs(
        self, 
        y_true, 
        y_pred, 
        filename: str
    ):
        """
        Plot predicted vs. observed (actual) values and save the plot.

        Args:
            y_true: The actual target values.
            y_pred: The predicted values from the model.
            output_dir: The directory to save the plot.
            filename: The name of the output PNG file (without extension).
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, edgecolors=(0, 0, 0))
        plt.plot(
            [min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2
            )
        plt.xlabel("Observed Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs. Observed Values")
        plt.grid(True)

        output_path = os.path.join(self.output_dir, f"{filename}.png")
        self.__save_plot(output_path)      

    def plot_learning_curve(
        self, 
        model, 
        X_train, 
        y_train, 
        cv: int = 5, 
        num_repeats: int = 1, 
        scoring: str = 'neg_mean_absolute_error', 
        filename: str = 'learning_curve'
    ):
        """
        Generate learning curves for a model and save them as a PNG file.

        Args:
            model: The machine learning model to evaluate.
            X_train (array-like): The input features of the training set.
            y_train (array-like): The target values of the training set.
            method_name (str): The name of the regression method.
            cv (int): Number of folds in cross-validation (default is 5).
            num_repeats (int): Number of times to repeat the cross-validation (default is 1).
            scoring (str): The scoring metric to use (default is 'neg_mean_absolute_error').
            output_dir (str): The directory to save the plot (default is './results').
            filename (str): The name of the output PNG file (default is 'learning_curve').
        """
        method_name = model.__class__.__name__

        cv = model_select.RepeatedKFold(n_splits=cv, n_repeats=num_repeats)

        # Generate learning curve data
        train_sizes, train_scores, test_scores, fit_times, _ = model_select.learning_curve(
            model, 
            X_train, 
            y_train, 
            cv=cv, 
            n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5), 
            return_times=True, 
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
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training Examples", fontsize=12)
        axes[1].set_ylabel("Fit Times", fontsize=12)
        axes[1].set_title("Scalability of the Model", fontsize=16)

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("Fit Times", fontsize=12)
        axes[2].set_ylabel(scoring, fontsize=12)
        axes[2].set_title("Performance of the Model", fontsize=16)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f"{filename}.png")
        self.__save_plot(output_path)

    def plot_feature_importance(
        self, 
        model, 
        X, 
        y, 
        filter, 
        feature_names, 
        filename, 
        scoring, 
        num_rep
    ) -> None:
        metric = self.scoring_config.get_scorer(scoring)


        if isinstance(model, (tree.DecisionTreeRegressor, ensemble.RandomForestRegressor, ensemble.GradientBoostingRegressor)):
            model.fit(X,y)
            importance = model.feature_importances_
        else:
            model.fit(X, y)
            results = inspection.permutation_importance(model, X=X, y=y, scoring=metric, n_repeats=num_rep)
            importance = results.importances_mean


        if isinstance(filter, int):
            sorted_indices = np.argsort(importance)[::-1]
            importance = importance[sorted_indices[:filter]]
            feature_names = [feature_names[i] for i in sorted_indices[:filter]]
        elif isinstance(filter, float):
            above_threshold = importance >= filter
            importance = importance[above_threshold]
            feature_names = [feature_names[i] for i in range(len(feature_names)) if above_threshold[i]]

        plt.barh(feature_names, importance)
        plt.xticks(rotation=90)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance', fontsize=16)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{filename}.png')
        self.__save_plot(output_path)

    def plot_residuals(self, model, X, y, filename):
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
        self.__save_plot(output_path)

    def hyperparameter_tuning(self, model, method, method_name, X_train, y_train, scorer, kf, num_rep, n_jobs) -> base.BaseEstimator:
        if method == "grid":
            searcher = model_select.GridSearchCV
        elif method == "random":
            searcher = model_select.RandomizedSearchCV
        else:
            raise ValueError(f"method must be one of (grid, random). {method} was entered.")
        
        metric = self.scoring_config.get_scorer(scorer)
        param_grid = self.method_config[method_name].get_hyperparam_grid()


        cv = model_select.RepeatedKFold(n_splits=kf, n_repeats=num_rep)
        search = searcher(
            estimator=model, param_grid=param_grid, n_jobs=n_jobs, cv=cv,
            scoring=metric
        )
        search_result = search.fit(X_train, y_train)
        tuned_model = self.method_config[method_name].instantiate_tuned(
            search_result.best_params_
            )
        tuned_model.fit(X_train, y_train)
        return tuned_model

    # Utility Methods
    def __save_to_json(self, data: Dict, output_path: str) -> None:
        """
        Save a dictionary to a JSON file.

        Args:
            data (dict): The data to save.
            output_path (str): The path to the output file.
        """
        try:
            with open(output_path, "w") as file:
                json.dump(data, file, indent=4)
        except IOError as e:
            print(f"Failed to save JSON to {output_path}: {e}") 

    def __save_plot(self, output_path: str) -> None:
        """
        Save the current matplotlib plot to a PNG file.

        Args:
            output_path (str): The full path (including filename) where the plot will be saved.
        """
        try:
            plt.savefig(output_path, format="png")
            plt.close()
        except IOError as e:
            print(f"Failed to save plot to {output_path}: {e}")

    def with_config(self, **kwargs):
        """
        Create a copy of the current evaluator and update with additional configuration.

        Args:
            kwargs: Key-value pairs of attributes to add/modify in the copy.
        
        Returns:
            ModelEvaluator: A copy of the current evaluator with updated attributes.
        """
        eval_copy = copy.copy(self)
        for key, value in kwargs.items():
            setattr(eval_copy, key, value)
        return eval_copy
