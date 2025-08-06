"""Common plots for evaluating models."""
from typing import Optional, Dict, Union, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotnine as pn
from sklearn import base
import sklearn.model_selection as model_select
from sklearn import inspection
from sklearn import tree
from sklearn import ensemble

from brisk.evaluation.evaluators.plot_evaluator import PlotEvaluator

class PlotLearningCurve(PlotEvaluator):
    """Plot learning curves showing model performance vs training size."""
    def plot(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
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
        X : DataFrame
            Training features
        y : Series
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
        plot_data = self._generate_plot_data(
            model, X, y, cv, num_repeats, n_jobs, metric
        )
        self._create_plot(plot_data, metric, model)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(filename, metadata)
        self._log_results("Learning Curve", filename)

    def _generate_plot_data(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
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
        X : DataFrame
            Training features
        y : Series
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
        splitter, indices = self.utility.get_cv_splitter(
            y, cv, num_repeats
        )
        results = {}
        scorer = self.metric_config.get_scorer(metric)

        # Generate learning curve data
        train_sizes, train_scores, test_scores, fit_times, _ = (
            model_select.learning_curve(
                model, X, y, cv=splitter, groups=indices,
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

    def _create_plot(
        self,
        results: Dict[str, float],
        metric: str,
        model: base.BaseEstimator
    ) -> None:
        # Create subplots
        _, axes = plt.subplots(1, 3, figsize=(16, 6))
        plt.rcParams.update({"font.size": 12})

        # Plot Learning Curve
        display_name = self.metric_config.get_name(metric)
        wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
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


class PlotFeatureImportance(PlotEvaluator):
    """Plot the feature importance for the model."""
    def plot(
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
        importance_data, plot_width, plot_height = self._generate_plot_data(
            model, X, y, threshold, feature_names, metric, num_rep
        )
        display_name = self.metric_config.get_name(metric)
        wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
        plot = self._create_plot(importance_data, display_name, wrapper)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(
            filename, metadata, plot=plot, height=plot_height, width=plot_width
        )
        self._log_results("Feature Importance", filename)

    def _generate_plot_data(
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
        importance_data["Feature"] = pd.Categorical(
            importance_data["Feature"],
            categories=importance_data.sort_values("Importance")["Feature"],
            ordered=True
        )
        return importance_data, plot_width, plot_height

    def _create_plot(
        self,
        importance_data: pd.DataFrame,
        display_name: str,
        wrapper: str
    ):
        plot = (
            pn.ggplot(importance_data, pn.aes(x="Feature", y="Importance")) +
            pn.geom_bar(stat="identity", fill=self.primary_color) +
            pn.coord_flip() +
            pn.labs(
                x="Feature", y=f"Importance ({display_name})",
                title=f"Feature Importance ({wrapper.display_name})"
            ) +
            self.theme.brisk_theme()
        )
        return plot


class PlotModelComparison(PlotEvaluator):
    """Plot a comparison of multiple models based on the specified measure."""
    def plot(
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
        plot_data = self._generate_plot_data(*models, X=X, y=y, metric=metric)
        if plot_data is None:
            return
        display_name = self.metric_config.get_name(metric)
        plot = self._create_plot(plot_data, display_name)
        metadata = self._generate_metadata(list(models), X.attrs["is_test"])
        self._save_plot(filename, metadata, plot=plot)
        self._log_results("Model Comparison", filename)

    def _generate_plot_data(
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
            wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
            model_names.append(wrapper.display_name)

        metric_values = []

        scorer = self.metric_config.get_metric(metric)

        for model in models:
            predictions = model.predict(X)
            if scorer is not None:
                score = scorer(y, predictions)
                metric_values.append(round(score, 3))
            else:
                self.services.logger.logger.info(f"Scorer for {metric} not found.")
                return None

        plot_data = pd.DataFrame({
            "Model": model_names,
            "Score": metric_values,
        })
        return plot_data

    def _create_plot(
        self,
        plot_data: pd.DataFrame,
        display_name: str
    ):
        plot = (
            pn.ggplot(plot_data, pn.aes(x="Model", y="Score")) +
            pn.geom_bar(stat="identity", fill=self.primary_color) +
            pn.geom_text(
                pn.aes(label="Score"), position=pn.position_stack(vjust=0.5),
                color="white", size=16
            ) +
            pn.ggtitle(f"Model Comparison on {display_name}") +
            pn.ylab(display_name) +
            self.theme.brisk_theme()
        )
        return plot
